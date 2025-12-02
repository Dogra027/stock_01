# agents/recommender_agent.py
"""
RecommenderAgent - LLM-driven reasoning agent for portfolio recommendations.

Combines signals, sentiment, market data, and filings to produce actionable
recommendations with evidence-based justification and confidence scores.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ========== DATA MODELS ==========

@dataclass
class ScoreComponents:
    """Individual score components for transparency and auditability."""
    news_impact_score: float  # -1.0 to 1.0
    technical_score: float    # -1.0 to 1.0
    portfolio_risk_score: float  # 0.0 to 1.0 (higher = riskier)
    sentiment_score: float    # -1.0 to 1.0
    filing_impact_score: float  # -1.0 to 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Recommendation:
    """Final recommendation output for a ticker."""
    ticker: str
    action: str  # "buy" | "sell" | "hold" | "monitor"
    confidence: float  # 0.0 to 1.0
    composite_score: float  # -1.0 to 1.0
    rationale: str  # Two-line natural language explanation
    evidence: List[str]  # Evidence IDs used
    score_components: ScoreComponents
    missing_data: List[str]  # List of missing inputs
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['score_components'] = self.score_components.to_dict()
        return data


# ========== SCORING FUNCTIONS ==========

def compute_news_impact_score(
    news_items: List[Dict[str, Any]],
    ticker: str
) -> tuple[float, List[str]]:
    """
    Compute news impact score for a ticker.
    
    Returns:
        (score, evidence_ids): Score from -1.0 to 1.0 and list of news IDs
    """
    if not news_items:
        return 0.0, []
    
    ticker_news = [n for n in news_items if ticker in n.get('tickers', [])]
    if not ticker_news:
        return 0.0, []
    
    # Weight by recency and source reliability
    total_score = 0.0
    total_weight = 0.0
    evidence_ids = []
    
    for news in ticker_news[:10]:  # Top 10 most recent
        # Simple sentiment from headline (you can enhance this)
        headline = news.get('headline', '').lower()
        
        # Basic sentiment keywords
        positive_words = ['surge', 'gain', 'profit', 'growth', 'beat', 'strong', 'up']
        negative_words = ['fall', 'loss', 'drop', 'weak', 'miss', 'down', 'decline']
        
        pos_count = sum(1 for word in positive_words if word in headline)
        neg_count = sum(1 for word in negative_words if word in headline)
        
        # Score: -1 to 1
        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            score = 0.0
        
        # Weight by source (you can make this more sophisticated)
        weight = 1.0
        
        total_score += score * weight
        total_weight += weight
        evidence_ids.append(news.get('id', f"news_{len(evidence_ids)}"))
    
    final_score = total_score / total_weight if total_weight > 0 else 0.0
    return final_score, evidence_ids


def compute_technical_score(
    market_data: Dict[str, Any],
    ticker: str
) -> tuple[float, List[str]]:
    """
    Compute technical score based on price movements.
    
    Returns:
        (score, evidence_ids): Score from -1.0 to 1.0 and evidence
    """
    if not market_data:
        return 0.0, []
    
    # Get price changes
    pct_1h = market_data.get('pct_change_1h', 0.0) or 0.0
    pct_24h = market_data.get('pct_change_24h', 0.0) or 0.0
    pct_7d = market_data.get('pct_change_7d', 0.0) or 0.0
    
    # Weighted average (more weight to recent)
    score = (pct_1h * 0.5 + pct_24h * 0.3 + pct_7d * 0.2) / 10.0  # Normalize to -1 to 1
    score = max(-1.0, min(1.0, score))  # Clamp
    
    evidence = [f"market_data_{ticker}"]
    return score, evidence


def compute_portfolio_risk_score(
    portfolio_json: List[Dict[str, Any]],
    ticker: str,
    market_data: Dict[str, Any]
) -> tuple[float, List[str]]:
    """
    Compute portfolio risk score (concentration risk).
    
    Returns:
        (score, evidence_ids): Score from 0.0 to 1.0 (higher = riskier)
    """
    if not portfolio_json:
        return 0.5, []  # Default medium risk
    
    # Find ticker in portfolio
    holding = next((h for h in portfolio_json if h.get('ticker') == ticker), None)
    if not holding:
        return 0.0, []
    
    # Calculate position size as % of portfolio
    position_pct = market_data.get('position_pct', 0.0) or 0.0
    
    # Risk increases with concentration
    # 0-5%: low risk (0.2)
    # 5-10%: medium risk (0.5)
    # 10-20%: high risk (0.8)
    # >20%: very high risk (1.0)
    if position_pct < 0.05:
        risk_score = 0.2
    elif position_pct < 0.10:
        risk_score = 0.5
    elif position_pct < 0.20:
        risk_score = 0.8
    else:
        risk_score = 1.0
    
    evidence = [f"portfolio_position_{ticker}"]
    return risk_score, evidence


def compute_sentiment_score(
    sentiment_data: Dict[str, Any],
    ticker: str
) -> tuple[float, List[str]]:
    """
    Extract sentiment score for ticker.
    
    Returns:
        (score, evidence_ids): Score from -1.0 to 1.0
    """
    if not sentiment_data:
        return 0.0, []
    
    ticker_sentiment = sentiment_data.get('ticker_sentiment', {}).get(ticker, {})
    if not ticker_sentiment:
        return 0.0, []
    
    score = ticker_sentiment.get('sentiment_score', 0.0)
    evidence = [f"sentiment_{ticker}"]
    
    return score, evidence


def compute_filing_impact_score(
    filings: List[Dict[str, Any]],
    ticker: str
) -> tuple[float, List[str]]:
    """
    Compute impact score from regulatory filings.
    
    Returns:
        (score, evidence_ids): Score from -1.0 to 1.0
    """
    if not filings:
        return 0.0, []
    
    ticker_filings = [f for f in filings if f.get('ticker') == ticker]
    if not ticker_filings:
        return 0.0, []
    
    # Map impact tags to scores
    impact_map = {
        'material': 0.7,
        'moderate': 0.3,
        'immaterial': 0.1
    }
    
    total_score = 0.0
    evidence_ids = []
    
    for filing in ticker_filings[:5]:  # Top 5 recent
        impact_tag = filing.get('impact_tag', 'immaterial')
        impact_value = impact_map.get(impact_tag, 0.1)
        
        # Determine if positive or negative (simple heuristic)
        filing_type = filing.get('type', '').upper()
        if '8-K' in filing_type or '10-Q' in filing_type:
            # Earnings reports - check if positive
            bullets = filing.get('bullets', [])
            positive_keywords = ['beat', 'exceed', 'strong', 'growth', 'profit']
            negative_keywords = ['miss', 'weak', 'decline', 'loss']
            
            pos_count = sum(1 for b in bullets for kw in positive_keywords if kw in b.lower())
            neg_count = sum(1 for b in bullets for kw in negative_keywords if kw in b.lower())
            
            if pos_count > neg_count:
                total_score += impact_value
            elif neg_count > pos_count:
                total_score -= impact_value
        
        evidence_ids.append(filing.get('id', f"filing_{len(evidence_ids)}"))
    
    # Normalize
    final_score = total_score / len(ticker_filings) if ticker_filings else 0.0
    final_score = max(-1.0, min(1.0, final_score))
    
    return final_score, evidence_ids


# ========== COMPOSITE SCORING ==========

def compute_composite_score(
    components: ScoreComponents,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Combine component scores into composite score.
    
    Args:
        components: Individual score components
        weights: Optional custom weights (default: balanced)
        
    Returns:
        Composite score from -1.0 to 1.0
    """
    if weights is None:
        weights = {
            'news_impact': 0.25,
            'technical': 0.20,
            'sentiment': 0.25,
            'filing_impact': 0.20,
            'portfolio_risk': -0.10  # Negative because high risk reduces score
        }
    
    composite = (
        components.news_impact_score * weights['news_impact'] +
        components.technical_score * weights['technical'] +
        components.sentiment_score * weights['sentiment'] +
        components.filing_impact_score * weights['filing_impact'] -
        components.portfolio_risk_score * weights['portfolio_risk']
    )
    
    return max(-1.0, min(1.0, composite))


def map_score_to_action(composite_score: float, missing_data: List[str]) -> str:
    """
    Map composite score to action label.
    
    Args:
        composite_score: Score from -1.0 to 1.0
        missing_data: List of missing inputs
        
    Returns:
        Action: "buy" | "sell" | "hold" | "monitor"
    """
    # Safety check: if critical data missing, default to monitor
    critical_missing = any(x in missing_data for x in ['market_data', 'news', 'sentiment'])
    if critical_missing:
        return "monitor"
    
    # Threshold mapping
    if composite_score >= 0.4:
        return "buy"
    elif composite_score <= -0.4:
        return "sell"
    elif -0.4 < composite_score < 0.4:
        return "hold"
    else:
        return "monitor"


# ========== LLM JUSTIFICATION ==========

def generate_llm_rationale(
    ticker: str,
    action: str,
    components: ScoreComponents,
    evidence_ids: List[str],
    missing_data: List[str]
) -> str:
    """
    Use LLM to generate natural language rationale.
    
    Args:
        ticker: Stock ticker
        action: Recommended action
        components: Score components
        evidence_ids: List of evidence IDs
        missing_data: List of missing inputs
        
    Returns:
        Two-line rationale
    """
    try:
        import requests
    except ImportError:
        return f"{action.upper()} {ticker}: Composite score {components.news_impact_score:.2f}. Based on available data."
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return f"{action.upper()} {ticker}: Composite score based on technical and sentiment analysis. Confidence moderate."
    
    # Structured prompt to avoid hallucination
    prompt = f"""Generate a TWO-LINE investment rationale for this recommendation. Use ONLY the provided evidence.

Ticker: {ticker}
Action: {action.upper()}

Evidence Scores:
- News Impact: {components.news_impact_score:.2f}
- Technical: {components.technical_score:.2f}
- Sentiment: {components.sentiment_score:.2f}
- Filing Impact: {components.filing_impact_score:.2f}
- Portfolio Risk: {components.portfolio_risk_score:.2f}

Evidence IDs: {', '.join(evidence_ids[:5])}
Missing Data: {', '.join(missing_data) if missing_data else 'None'}

Rules:
1. EXACTLY two lines
2. First line: Main reason for {action}
3. Second line: Supporting factor or risk consideration
4. DO NOT invent facts
5. Reference only the scores provided
6. Be specific and actionable

Rationale:"""

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        rationale = result["choices"][0]["message"]["content"].strip()
        
        # Ensure it's two lines
        lines = [l.strip() for l in rationale.split('\n') if l.strip()]
        if len(lines) >= 2:
            return f"{lines[0]}\n{lines[1]}"
        else:
            return rationale
            
    except Exception as e:
        print(f"LLM rationale generation failed: {e}")
        return f"{action.upper()} {ticker}: Based on composite analysis of available signals. Review recommended."


# ========== MAIN AGENT ==========

class RecommenderAgent:
    """
    LLM-driven recommendation agent with evidence-based reasoning.
    """
    
    def __init__(
        self,
        score_weights: Optional[Dict[str, float]] = None,
        use_llm_rationale: bool = True
    ):
        """
        Initialize RecommenderAgent.
        
        Args:
            score_weights: Custom weights for composite scoring
            use_llm_rationale: Whether to use LLM for rationale generation
        """
        self.score_weights = score_weights
        self.use_llm_rationale = use_llm_rationale
    
    def run(
        self,
        portfolio_json: List[Dict[str, Any]],
        market_data: Dict[str, List[Dict[str, Any]]],
        signals: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        filings: Optional[List[Dict[str, Any]]] = None,
        news_items: Optional[List[Dict[str, Any]]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for portfolio tickers.
        
        Args:
            portfolio_json: Portfolio holdings
            market_data: Market data by ticker
            signals: Trading signals (optional)
            sentiment: Sentiment analysis results (optional)
            filings: Regulatory filings (optional)
            news_items: News articles (optional)
            user_preferences: User preferences (optional)
            
        Returns:
            Dict with 'recommendations' list
        """
        recommendations = []
        
        # Extract tickers from portfolio
        tickers = list(set(h.get('ticker') for h in portfolio_json if h.get('ticker')))
        
        for ticker in tickers:
            # Track missing data
            missing_data = []
            all_evidence = []
            
            # Get ticker-specific market data
            ticker_market_data = next(
                (md for md in market_data if md.get('ticker') == ticker),
                {}
            ) if isinstance(market_data, list) else market_data.get(ticker, {})
            
            if not ticker_market_data:
                missing_data.append('market_data')
            
            # Compute individual scores
            news_score, news_evidence = compute_news_impact_score(
                news_items or [], ticker
            )
            if not news_items:
                missing_data.append('news')
            all_evidence.extend(news_evidence)
            
            tech_score, tech_evidence = compute_technical_score(
                ticker_market_data, ticker
            )
            all_evidence.extend(tech_evidence)
            
            risk_score, risk_evidence = compute_portfolio_risk_score(
                portfolio_json, ticker, ticker_market_data
            )
            all_evidence.extend(risk_evidence)
            
            sent_score, sent_evidence = compute_sentiment_score(
                sentiment or {}, ticker
            )
            if not sentiment:
                missing_data.append('sentiment')
            all_evidence.extend(sent_evidence)
            
            filing_score, filing_evidence = compute_filing_impact_score(
                filings or [], ticker
            )
            if not filings:
                missing_data.append('filings')
            all_evidence.extend(filing_evidence)
            
            # Create score components
            components = ScoreComponents(
                news_impact_score=news_score,
                technical_score=tech_score,
                portfolio_risk_score=risk_score,
                sentiment_score=sent_score,
                filing_impact_score=filing_score
            )
            
            # Compute composite score
            composite = compute_composite_score(components, self.score_weights)
            
            # Map to action
            action = map_score_to_action(composite, missing_data)
            
            # Generate rationale
            if self.use_llm_rationale:
                rationale = generate_llm_rationale(
                    ticker, action, components, all_evidence, missing_data
                )
            else:
                rationale = f"{action.upper()} based on composite score {composite:.2f}.\nReview individual components for details."
            
            # Calculate confidence (inverse of missing data + score certainty)
            confidence = 1.0 - (len(missing_data) * 0.15)  # Reduce by 15% per missing input
            confidence = max(0.0, min(1.0, confidence))
            
            # Create recommendation
            rec = Recommendation(
                ticker=ticker,
                action=action,
                confidence=confidence,
                composite_score=composite,
                rationale=rationale,
                evidence=all_evidence[:10],  # Top 10 evidence items
                score_components=components,
                missing_data=missing_data,
                timestamp=datetime.now().isoformat()
            )
            
            recommendations.append(rec.to_dict())
        
        return {
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'total_tickers': len(tickers)
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Example data
    portfolio = [
        {"ticker": "AAPL", "qty": 100, "holding_value": 15000},
        {"ticker": "GOOGL", "qty": 50, "holding_value": 7500}
    ]
    
    market_data = [
        {
            "ticker": "AAPL",
            "price": 150.0,
            "pct_change_1h": 0.5,
            "pct_change_24h": 2.0,
            "pct_change_7d": 5.0,
            "position_pct": 0.67
        },
        {
            "ticker": "GOOGL",
            "price": 150.0,
            "pct_change_1h": -0.3,
            "pct_change_24h": -1.0,
            "pct_change_7d": -2.0,
            "position_pct": 0.33
        }
    ]
    
    sentiment_data = {
        "ticker_sentiment": {
            "AAPL": {"sentiment_score": 0.45, "confidence": 0.8},
            "GOOGL": {"sentiment_score": -0.2, "confidence": 0.6}
        }
    }
    
    news = [
        {"id": "n1", "tickers": ["AAPL"], "headline": "Apple reports strong earnings growth"},
        {"id": "n2", "tickers": ["GOOGL"], "headline": "Google faces regulatory challenges"}
    ]
    
    # Run agent
    agent = RecommenderAgent(use_llm_rationale=True)
    results = agent.run(
        portfolio_json=portfolio,
        market_data=market_data,
        sentiment=sentiment_data,
        news_items=news
    )
    
    # Display results
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(json.dumps(results, indent=2))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for rec in results['recommendations']:
        print(f"\n{rec['ticker']}: {rec['action'].upper()} (Confidence: {rec['confidence']:.2f})")
        print(f"Composite Score: {rec['composite_score']:.3f}")
        print(f"Rationale:\n  {rec['rationale']}")
        if rec['missing_data']:
            print(f"Missing: {', '.join(rec['missing_data'])}")
