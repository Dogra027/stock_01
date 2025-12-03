# agents/explainability_agent.py
"""
ExplainabilityAgent - LLM-assisted summarizer and audit logger.

Produces human-friendly rationales with evidence traceback and maintains
an audit trail for all recommendations.
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
class EvidenceItem:
    """Individual piece of evidence."""
    id: str
    type: str  # "signal" | "news" | "filing" | "metric"
    description: str
    value: Optional[float] = None
    impact: Optional[str] = None  # "positive" | "negative" | "neutral"


@dataclass
class AuditRecord:
    """Audit trail record for a recommendation."""
    run_id: str
    ticker: str
    timestamp: str
    action: str
    composite_score: float
    score_components: Dict[str, float]
    evidence_ids: List[str]
    confidence: float
    missing_data: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Explanation:
    """Human-friendly explanation for a recommendation."""
    ticker: str
    action: str
    rationale_bullets: List[str]  # Bullet points explaining the decision
    top_evidence: List[EvidenceItem]  # Top contributing evidence
    score_breakdown: Dict[str, float]  # Component scores
    audit_record: AuditRecord
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['top_evidence'] = [e.__dict__ if hasattr(e, '__dict__') else e for e in self.top_evidence]
        data['audit_record'] = self.audit_record.to_dict()
        return data


# ========== EVIDENCE TRACEBACK ==========

def extract_evidence_from_signals(
    signals: List[Dict[str, Any]],
    ticker: str
) -> List[EvidenceItem]:
    """Extract evidence items from signals."""
    evidence = []
    
    if not signals:
        return evidence
    
    ticker_signals = [s for s in signals if s.get('ticker') == ticker]
    
    for signal in ticker_signals[:5]:  # Top 5 signals
        impact = "positive" if "spike" in signal.get('type', '') or "positive" in signal.get('type', '') else \
                 "negative" if "drop" in signal.get('type', '') or "negative" in signal.get('type', '') else \
                 "neutral"
        
        evidence.append(EvidenceItem(
            id=f"signal_{signal.get('type', 'unknown')}",
            type="signal",
            description=signal.get('explanation', 'Signal detected'),
            value=signal.get('severity'),
            impact=impact
        ))
    
    return evidence


def extract_evidence_from_news(
    news_items: List[Dict[str, Any]],
    ticker: str
) -> List[EvidenceItem]:
    """Extract evidence items from news."""
    evidence = []
    
    if not news_items:
        return evidence
    
    ticker_news = [n for n in news_items if ticker in n.get('tickers', [])]
    
    for news in ticker_news[:3]:  # Top 3 news items
        headline = news.get('headline', 'News article')
        
        # Determine impact from headline
        positive_words = ['surge', 'gain', 'profit', 'growth', 'beat', 'strong']
        negative_words = ['fall', 'loss', 'drop', 'weak', 'miss', 'decline']
        
        headline_lower = headline.lower()
        has_positive = any(word in headline_lower for word in positive_words)
        has_negative = any(word in headline_lower for word in negative_words)
        
        impact = "positive" if has_positive else "negative" if has_negative else "neutral"
        
        evidence.append(EvidenceItem(
            id=news.get('id', f"news_{len(evidence)}"),
            type="news",
            description=f"{news.get('source', 'News')}: {headline[:80]}...",
            impact=impact
        ))
    
    return evidence


def extract_evidence_from_filings(
    filings: List[Dict[str, Any]],
    ticker: str
) -> List[EvidenceItem]:
    """Extract evidence items from filings."""
    evidence = []
    
    if not filings:
        return evidence
    
    ticker_filings = [f for f in filings if f.get('ticker') == ticker]
    
    for filing in ticker_filings[:2]:  # Top 2 filings
        filing_type = filing.get('type', 'Filing')
        summary = filing.get('short_summary', 'Regulatory filing')
        impact_tag = filing.get('impact_tag', 'immaterial')
        
        impact = "positive" if impact_tag == "material" else "neutral"
        
        evidence.append(EvidenceItem(
            id=filing.get('id', f"filing_{len(evidence)}"),
            type="filing",
            description=f"{filing_type}: {summary[:60]}...",
            impact=impact
        ))
    
    return evidence


def extract_evidence_from_metrics(
    recommendation: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None
) -> List[EvidenceItem]:
    """Extract evidence from metrics and scores."""
    evidence = []
    
    components = recommendation.get('score_components', {})
    
    # Technical metrics
    if market_data:
        pct_1h = market_data.get('pct_change_1h')
        if pct_1h is not None:
            impact = "positive" if pct_1h > 0 else "negative" if pct_1h < 0 else "neutral"
            evidence.append(EvidenceItem(
                id="metric_price_1h",
                type="metric",
                description=f"1-hour price change: {pct_1h:+.2f}%",
                value=pct_1h,
                impact=impact
            ))
        
        pct_24h = market_data.get('pct_change_24h')
        if pct_24h is not None:
            impact = "positive" if pct_24h > 0 else "negative" if pct_24h < 0 else "neutral"
            evidence.append(EvidenceItem(
                id="metric_price_24h",
                type="metric",
                description=f"24-hour price change: {pct_24h:+.2f}%",
                value=pct_24h,
                impact=impact
            ))
    
    # Sentiment score
    sent_score = components.get('sentiment_score')
    if sent_score is not None and sent_score != 0:
        impact = "positive" if sent_score > 0 else "negative"
        evidence.append(EvidenceItem(
            id="metric_sentiment",
            type="metric",
            description=f"Sentiment score: {sent_score:+.2f}",
            value=sent_score,
            impact=impact
        ))
    
    return evidence


# ========== RATIONALE GENERATION ==========

def generate_bullet_rationale(
    recommendation: Dict[str, Any],
    evidence_items: List[EvidenceItem]
) -> List[str]:
    """Generate bullet-point rationale from evidence."""
    bullets = []
    
    action = recommendation.get('action', 'hold').upper()
    ticker = recommendation.get('ticker', 'UNKNOWN')
    composite_score = recommendation.get('composite_score', 0.0)
    components = recommendation.get('score_components', {})
    
    # Main recommendation bullet
    bullets.append(f"**{action} {ticker}** (Composite Score: {composite_score:+.2f})")
    
    # Score components breakdown
    score_bullets = []
    if components.get('news_impact_score', 0) != 0:
        score_bullets.append(f"News Impact: {components['news_impact_score']:+.2f}")
    if components.get('technical_score', 0) != 0:
        score_bullets.append(f"Technical: {components['technical_score']:+.2f}")
    if components.get('sentiment_score', 0) != 0:
        score_bullets.append(f"Sentiment: {components['sentiment_score']:+.2f}")
    if components.get('filing_impact_score', 0) != 0:
        score_bullets.append(f"Filing Impact: {components['filing_impact_score']:+.2f}")
    if components.get('portfolio_risk_score', 0) != 0:
        score_bullets.append(f"Portfolio Risk: {components['portfolio_risk_score']:.2f}")
    
    if score_bullets:
        bullets.append("Score Components: " + ", ".join(score_bullets))
    
    # Top evidence items
    positive_evidence = [e for e in evidence_items if e.impact == "positive"]
    negative_evidence = [e for e in evidence_items if e.impact == "negative"]
    
    if positive_evidence:
        bullets.append("**Positive Factors:**")
        for ev in positive_evidence[:3]:
            bullets.append(f"  • {ev.description}")
    
    if negative_evidence:
        bullets.append("**Negative Factors:**")
        for ev in negative_evidence[:3]:
            bullets.append(f"  • {ev.description}")
    
    # Missing data warning
    missing_data = recommendation.get('missing_data', [])
    if missing_data:
        bullets.append(f"⚠️ Missing data: {', '.join(missing_data)}")
    
    return bullets


def generate_plain_english_summary(
    recommendation: Dict[str, Any],
    evidence_items: List[EvidenceItem]
) -> str:
    """Generate a plain-English summary referencing specific evidence."""
    ticker = recommendation.get('ticker', 'UNKNOWN')
    action = recommendation.get('action', 'hold').upper()
    
    # Collect specific evidence mentions
    evidence_mentions = []
    
    for ev in evidence_items[:5]:  # Top 5 pieces of evidence
        if ev.type == "metric" and ev.value is not None:
            evidence_mentions.append(ev.description)
        elif ev.type == "news":
            # Extract source and headline snippet
            evidence_mentions.append(ev.description)
        elif ev.type == "filing":
            evidence_mentions.append(ev.description)
        elif ev.type == "signal":
            evidence_mentions.append(ev.description)
    
    if evidence_mentions:
        evidence_text = "; ".join(evidence_mentions[:3])  # Top 3
        summary = f"{action} {ticker} based on: {evidence_text}"
    else:
        summary = f"{action} {ticker} based on composite analysis"
    
    return summary


# ========== AUDIT LOGGING ==========

def create_audit_record(
    run_id: str,
    recommendation: Dict[str, Any]
) -> AuditRecord:
    """Create an audit record for a recommendation."""
    return AuditRecord(
        run_id=run_id,
        ticker=recommendation.get('ticker', 'UNKNOWN'),
        timestamp=recommendation.get('timestamp', datetime.now().isoformat()),
        action=recommendation.get('action', 'hold'),
        composite_score=recommendation.get('composite_score', 0.0),
        score_components=recommendation.get('score_components', {}),
        evidence_ids=recommendation.get('evidence', []),
        confidence=recommendation.get('confidence', 0.0),
        missing_data=recommendation.get('missing_data', [])
    )


def save_audit_log(
    audit_records: List[AuditRecord],
    run_id: str,
    persist: bool = False
) -> str:
    """Save audit log (ephemeral by default)."""
    log_data = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'total_recommendations': len(audit_records),
        'records': [r.to_dict() for r in audit_records]
    }
    
    if persist:
        # Save to file (in production, this would be a database)
        log_file = f"/tmp/audit_log_{run_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        return log_file
    else:
        # Ephemeral - just return as string
        return json.dumps(log_data, indent=2)


# ========== MAIN AGENT ==========

class ExplainabilityAgent:
    """
    LLM-assisted summarizer and audit logger for recommendations.
    """
    
    def __init__(self, persist_audit: bool = False):
        """
        Initialize ExplainabilityAgent.
        
        Args:
            persist_audit: Whether to persist audit logs to disk (default: False, ephemeral)
        """
        self.persist_audit = persist_audit
    
    def run(
        self,
        recommendations: List[Dict[str, Any]],
        signals: Optional[List[Dict[str, Any]]] = None,
        news_items: Optional[List[Dict[str, Any]]] = None,
        filings: Optional[List[Dict[str, Any]]] = None,
        market_data: Optional[List[Dict[str, Any]]] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate explanations and audit trail for recommendations.
        
        Args:
            recommendations: List of recommendations from RecommenderAgent
            signals: Optional trading signals
            news_items: Optional news articles
            filings: Optional regulatory filings
            market_data: Optional market data
            run_id: Optional run identifier
            
        Returns:
            Dict with 'explanations' list and 'audit_log'
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        explanations = []
        audit_records = []
        
        for rec in recommendations:
            ticker = rec.get('ticker', 'UNKNOWN')
            
            # Get ticker-specific market data
            ticker_market_data = None
            if market_data:
                ticker_market_data = next(
                    (md for md in market_data if md.get('ticker') == ticker),
                    None
                )
            
            # Extract evidence from all sources
            all_evidence = []
            
            # Signals
            if signals:
                all_evidence.extend(extract_evidence_from_signals(signals, ticker))
            
            # News
            if news_items:
                all_evidence.extend(extract_evidence_from_news(news_items, ticker))
            
            # Filings
            if filings:
                all_evidence.extend(extract_evidence_from_filings(filings, ticker))
            
            # Metrics
            all_evidence.extend(extract_evidence_from_metrics(rec, ticker_market_data))
            
            # Sort by impact (positive first, then negative, then neutral)
            impact_order = {"positive": 0, "negative": 1, "neutral": 2}
            all_evidence.sort(key=lambda e: impact_order.get(e.impact, 3))
            
            # Generate rationale bullets
            rationale_bullets = generate_bullet_rationale(rec, all_evidence)
            
            # Create audit record
            audit_record = create_audit_record(run_id, rec)
            audit_records.append(audit_record)
            
            # Create explanation
            explanation = Explanation(
                ticker=ticker,
                action=rec.get('action', 'hold'),
                rationale_bullets=rationale_bullets,
                top_evidence=all_evidence[:10],  # Top 10 evidence items
                score_breakdown=rec.get('score_components', {}),
                audit_record=audit_record,
                timestamp=datetime.now().isoformat()
            )
            
            explanations.append(explanation.to_dict())
        
        # Save audit log
        audit_log = save_audit_log(audit_records, run_id, self.persist_audit)
        
        return {
            'explanations': explanations,
            'audit_log': audit_log,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'total_explanations': len(explanations)
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Example recommendation
    recommendations = [
        {
            'ticker': 'AAPL',
            'action': 'buy',
            'confidence': 0.85,
            'composite_score': 0.499,
            'score_components': {
                'news_impact_score': 1.0,
                'technical_score': 0.185,
                'portfolio_risk_score': 1.0,
                'sentiment_score': 0.45,
                'filing_impact_score': 0.0
            },
            'evidence': ['n1', 'market_data_AAPL', 'sentiment_AAPL'],
            'missing_data': ['filings'],
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    signals = [
        {
            'ticker': 'AAPL',
            'type': 'price_drop_1h',
            'severity': 0.42,
            'explanation': 'Price dropped 2.50% in last hour'
        },
        {
            'ticker': 'AAPL',
            'type': 'unusual_volume',
            'severity': 0.17,
            'explanation': 'Volume 4.0x average'
        }
    ]
    
    news_items = [
        {
            'id': 'n1',
            'tickers': ['AAPL'],
            'headline': 'Apple reports strong earnings growth',
            'source': 'Reuters'
        }
    ]
    
    market_data = [
        {
            'ticker': 'AAPL',
            'pct_change_1h': -2.5,
            'pct_change_24h': 3.0
        }
    ]
    
    # Run agent
    agent = ExplainabilityAgent(persist_audit=False)
    results = agent.run(
        recommendations=recommendations,
        signals=signals,
        news_items=news_items,
        market_data=market_data
    )
    
    # Display results
    print("="*70)
    print("EXPLANATIONS")
    print("="*70)
    
    for exp in results['explanations']:
        print(f"\n{exp['ticker']}: {exp['action'].upper()}")
        print("\nRationale:")
        for bullet in exp['rationale_bullets']:
            print(f"  {bullet}")
        
        print(f"\nTop Evidence ({len(exp['top_evidence'])} items):")
        for i, ev in enumerate(exp['top_evidence'][:5], 1):
            impact_icon = "✅" if ev['impact'] == "positive" else "❌" if ev['impact'] == "negative" else "➖"
            print(f"  {i}. {impact_icon} {ev['description']}")
    
    print("\n" + "="*70)
    print("AUDIT LOG")
    print("="*70)
    print(results['audit_log'][:500] + "...")  # First 500 chars
