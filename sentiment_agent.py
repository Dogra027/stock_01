"""
SentimentAgent - NLP-based sentiment scoring for news & social media.

Pattern: Stateless microservice (NLP pipeline)
Primary goal: Score news & social sentiment, highlight evidence sentences

Inputs:
    - news_items: List[Dict] with keys: text, ticker, timestamp, source, cluster_id (optional)
    - social_posts: Optional[List[Dict]] with same structure
    - weighting config: news_weight vs social_weight

Outputs:
    - sentiment_scores per ticker
    - sentiment_scores per cluster (if cluster_id provided)
    - top evidence sentences
    - confidence scores

Work steps:
    1. Preprocess text (cleaning, remove boilerplate)
    2. Sentence-level scoring: run sentiment model
    3. Aggregate to article-level and ticker-level using time decay and source weighting
    4. Confidence: compute based on length, source reliability, score variance
    5. Return structured results

Notes:
    - Uses VADER for fast, real-time sentiment scoring
    - LLM can be added later for deeper summarization
"""

from __future__ import annotations

import re
import json
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


# ========== PREPROCESSING ==========

def preprocess_text(text: str) -> List[str]:
    """
    Clean text, split into sentences, remove boilerplate.
    
    Args:
        text: Raw text content
        
    Returns:
        List of cleaned sentences
    """
    if not text:
        return []
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())
    
    # Remove common boilerplate patterns
    boilerplate_patterns = [
        r"Read more at.*",
        r"For more info.*",
        r"Advertisement.*",
        r"Click here.*",
        r"Subscribe to.*",
        r"Follow us on.*",
        r"Share this article.*",
        r"Copyright \d{4}.*",
        r"All rights reserved.*",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Split into sentences (improved regex for better sentence boundary detection)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    
    # Filter out very short sentences (likely noise)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences


# ========== SENTENCE-LEVEL SCORING ==========

def score_sentences(sentences: List[str], analyzer=None) -> List[Dict[str, Any]]:
    """
    Run sentiment model on each sentence.
    
    Args:
        sentences: List of sentence strings
        analyzer: VADER SentimentIntensityAnalyzer instance
        
    Returns:
        List of dicts with 'sentence', 'score', 'pos', 'neg', 'neu'
    """
    results = []
    for s in sentences:
        if analyzer:
            scores = analyzer.polarity_scores(s)
            results.append({
                "sentence": s,
                "score": scores['compound'],  # -1 to +1
                "pos": scores['pos'],
                "neg": scores['neg'],
                "neu": scores['neu'],
            })
        else:
            # Fallback if VADER not available
            results.append({
                "sentence": s,
                "score": 0.0,
                "pos": 0.0,
                "neg": 0.0,
                "neu": 1.0,
            })
    return results


# ========== AGGREGATION ==========

def aggregate_scores(
    items: List[Dict[str, Any]],
    weight: float = 1.0,
    time_decay: float = 0.95
) -> Dict[str, Any]:
    """
    Aggregate sentence scores using time decay and weighting.
    
    Args:
        items: List of scored items with 'score', 'timestamp', 'text'
        weight: Base weight for this source type
        time_decay: Decay factor per day (0-1)
        
    Returns:
        Dict with 'sentiment_score', 'top_evidence', 'variance'
    """
    if not items:
        return {
            "sentiment_score": 0.0,
            "top_evidence": [],
            "variance": 0.0,
            "count": 0
        }
    
    now = datetime.now(timezone.utc)
    total_score = 0.0
    total_weight = 0.0
    evidence = []
    scores_list = []
    
    for item in items:
        ts = item.get("timestamp")
        score = item.get("score", 0.0)
        text = item.get("text", "")
        
        # Calculate time decay
        if ts:
            try:
                if isinstance(ts, str):
                    ts_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                else:
                    ts_dt = ts
                age_days = (now - ts_dt).days
                decay = time_decay ** max(0, age_days)
            except (ValueError, TypeError):
                decay = 1.0
        else:
            decay = 1.0
        
        # Apply weight and decay
        w = weight * decay
        total_score += score * w
        total_weight += w
        scores_list.append(score)
        
        evidence.append({
            "sentence": text,
            "score": score,
            "weight": w
        })
    
    # Calculate average score
    avg_score = total_score / total_weight if total_weight > 0 else 0.0
    
    # Calculate variance for confidence scoring
    variance = statistics.variance(scores_list) if len(scores_list) > 1 else 0.0
    
    # Get top evidence sentences (highest absolute scores)
    top_evidence = sorted(
        evidence,
        key=lambda x: abs(x["score"]),
        reverse=True
    )[:5]  # Top 5 evidence sentences
    
    return {
        "sentiment_score": avg_score,
        "top_evidence": [{"sentence": e["sentence"], "score": e["score"]} for e in top_evidence],
        "variance": variance,
        "count": len(items)
    }


# ========== CONFIDENCE SCORING ==========

def compute_confidence(
    items: List[Dict[str, Any]],
    source_reliability: float = 1.0,
    variance: float = 0.0
) -> float:
    """
    Compute confidence score based on multiple factors.
    
    Factors:
        - Text length (more text = higher confidence)
        - Source reliability (news > social)
        - Score variance (lower variance = higher confidence)
        - Number of items (more items = higher confidence)
    
    Args:
        items: List of items to compute confidence for
        source_reliability: Reliability factor (0-1)
        variance: Score variance
        
    Returns:
        Confidence score (0-1)
    """
    if not items:
        return 0.0
    
    # Factor 1: Text length
    total_length = sum(len(item.get("text", "")) for item in items)
    avg_length = total_length / len(items)
    length_factor = min(1.0, avg_length / 200)  # Normalize to 200 chars
    
    # Factor 2: Source reliability (passed in)
    reliability_factor = source_reliability
    
    # Factor 3: Variance (lower is better)
    variance_factor = 1.0 / (1.0 + variance) if variance > 0 else 1.0
    
    # Factor 4: Sample size
    count_factor = min(1.0, len(items) / 10)  # Normalize to 10 items
    
    # Weighted combination
    confidence = (
        0.3 * length_factor +
        0.3 * reliability_factor +
        0.2 * variance_factor +
        0.2 * count_factor
    )
    
    return min(1.0, max(0.0, confidence))


# ========== SOURCE RELIABILITY ==========

SOURCE_RELIABILITY = {
    "news": 1.0,
    "social": 0.6,
    "blog": 0.7,
    "press_release": 0.8,
    "unknown": 0.5,
}


def get_source_reliability(source: str) -> float:
    """Get reliability score for a source type."""
    return SOURCE_RELIABILITY.get(source.lower(), SOURCE_RELIABILITY["unknown"])


# ========== MAIN AGENT ==========

class SentimentAgent:
    """
    SentimentAgent - Stateless NLP microservice for sentiment analysis.
    
    Features:
        - Sentence-level sentiment scoring using VADER
        - Ticker-level and cluster-level aggregation
        - Time decay for recency weighting
        - Source reliability weighting
        - Confidence scoring
        - Top evidence sentence extraction
    """
    
    def __init__(
        self,
        news_weight: float = 1.0,
        social_weight: float = 0.5,
        time_decay: float = 0.95
    ):
        """
        Initialize SentimentAgent.
        
        Args:
            news_weight: Weight for news items (default: 1.0)
            social_weight: Weight for social posts (default: 0.5)
            time_decay: Daily decay factor for time weighting (default: 0.95)
        """
        self.news_weight = news_weight
        self.social_weight = social_weight
        self.time_decay = time_decay
        
        # Initialize VADER analyzer
        if SentimentIntensityAnalyzer:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
            print("WARNING: vaderSentiment not installed. Sentiment scores will be 0.")
    
    def run(
        self,
        news_items: List[Dict[str, Any]],
        social_posts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run sentiment analysis on news and social posts.
        
        Args:
            news_items: List of news items with keys:
                - text: str (required)
                - ticker: str (required)
                - timestamp: str (optional, format: "YYYY-MM-DD HH:MM:SS")
                - source: str (optional)
                - cluster_id: str (optional)
            social_posts: Optional list of social posts (same structure)
            
        Returns:
            Dict with keys:
                - ticker_sentiment: Dict[ticker -> sentiment_data]
                - cluster_sentiment: Dict[cluster_id -> sentiment_data]
                - overall_sentiment: Overall sentiment data
        """
        all_items = []
        
        # Process news items
        for item in news_items:
            sentences = preprocess_text(item.get("text", ""))
            sent_scores = score_sentences(sentences, self.analyzer)
            
            for s in sent_scores:
                all_items.append({
                    "ticker": item.get("ticker", "UNKNOWN"),
                    "cluster_id": item.get("cluster_id"),
                    "score": s["score"],
                    "text": s["sentence"],
                    "timestamp": item.get("timestamp"),
                    "source": item.get("source", "news"),
                    "weight": self.news_weight,
                })
        
        # Process social posts
        if social_posts:
            for item in social_posts:
                sentences = preprocess_text(item.get("text", ""))
                sent_scores = score_sentences(sentences, self.analyzer)
                
                for s in sent_scores:
                    all_items.append({
                        "ticker": item.get("ticker", "UNKNOWN"),
                        "cluster_id": item.get("cluster_id"),
                        "score": s["score"],
                        "text": s["sentence"],
                        "timestamp": item.get("timestamp"),
                        "source": item.get("source", "social"),
                        "weight": self.social_weight,
                    })
        
        # Aggregate by ticker
        ticker_map = defaultdict(list)
        for item in all_items:
            ticker_map[item["ticker"]].append(item)
        
        ticker_sentiment = {}
        for ticker, items in ticker_map.items():
            # Get source reliability (use first item's source as representative)
            source = items[0]["source"]
            reliability = get_source_reliability(source)
            
            # Aggregate scores
            agg = aggregate_scores(items, weight=items[0]["weight"], time_decay=self.time_decay)
            
            # Compute confidence
            conf = compute_confidence(items, source_reliability=reliability, variance=agg["variance"])
            
            ticker_sentiment[ticker] = {
                "sentiment_score": agg["sentiment_score"],
                "top_evidence": agg["top_evidence"],
                "confidence": conf,
                "count": agg["count"],
                "variance": agg["variance"],
            }
        
        # Aggregate by cluster (if cluster_id provided)
        cluster_map = defaultdict(list)
        for item in all_items:
            if item.get("cluster_id"):
                cluster_map[item["cluster_id"]].append(item)
        
        cluster_sentiment = {}
        for cluster_id, items in cluster_map.items():
            source = items[0]["source"]
            reliability = get_source_reliability(source)
            
            agg = aggregate_scores(items, weight=items[0]["weight"], time_decay=self.time_decay)
            conf = compute_confidence(items, source_reliability=reliability, variance=agg["variance"])
            
            cluster_sentiment[cluster_id] = {
                "sentiment_score": agg["sentiment_score"],
                "top_evidence": agg["top_evidence"],
                "confidence": conf,
                "count": agg["count"],
                "variance": agg["variance"],
            }
        
        # Overall sentiment
        if all_items:
            overall_agg = aggregate_scores(all_items, time_decay=self.time_decay)
            overall_conf = compute_confidence(all_items, source_reliability=0.8, variance=overall_agg["variance"])
            overall_sentiment = {
                "sentiment_score": overall_agg["sentiment_score"],
                "top_evidence": overall_agg["top_evidence"],
                "confidence": overall_conf,
                "count": overall_agg["count"],
            }
        else:
            overall_sentiment = {
                "sentiment_score": 0.0,
                "top_evidence": [],
                "confidence": 0.0,
                "count": 0,
            }
        
        return {
            "ticker_sentiment": ticker_sentiment,
            "cluster_sentiment": cluster_sentiment,
            "overall_sentiment": overall_sentiment,
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Example news and social data
    news = [
        {
            "ticker": "AAPL",
            "text": "Apple reported strong earnings. Revenue up 10%. CEO optimistic about future growth.",
            "timestamp": "2025-11-24 10:00:00",
            "source": "news",
            "cluster_id": "earnings_report"
        },
        {
            "ticker": "AAPL",
            "text": "Apple faces lawsuit over patents. Legal experts say case is weak.",
            "timestamp": "2025-11-23 09:00:00",
            "source": "news",
            "cluster_id": "legal_issues"
        },
        {
            "ticker": "GOOGL",
            "text": "Google announces new AI breakthrough. Investors excited about potential.",
            "timestamp": "2025-11-24 11:30:00",
            "source": "news",
            "cluster_id": "ai_innovation"
        },
    ]
    
    social = [
        {
            "ticker": "AAPL",
            "text": "$AAPL to the moon! Great earnings!",
            "timestamp": "2025-11-24 11:00:00",
            "source": "social",
            "cluster_id": "earnings_report"
        },
        {
            "ticker": "GOOGL",
            "text": "$GOOGL AI is game-changing. Bullish!",
            "timestamp": "2025-11-24 12:00:00",
            "source": "social",
            "cluster_id": "ai_innovation"
        },
    ]
    
    # Initialize and run agent
    agent = SentimentAgent(news_weight=1.0, social_weight=0.5, time_decay=0.95)
    results = agent.run(news, social)
    
    # Pretty print results
    print("=" * 60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("\n" + "=" * 60)
    print("TICKER SENTIMENT SUMMARY")
    print("=" * 60)
    for ticker, data in results["ticker_sentiment"].items():
        print(f"\n{ticker}:")
        print(f"  Score: {data['sentiment_score']:.3f}")
        print(f"  Confidence: {data['confidence']:.3f}")
        print(f"  Count: {data['count']}")
        print(f"  Top Evidence:")
        for i, ev in enumerate(data['top_evidence'][:3], 1):
            print(f"    {i}. [{ev['score']:+.3f}] {ev['sentence'][:80]}...")

