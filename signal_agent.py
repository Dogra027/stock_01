# agents/signal_agent.py
"""
SignalAgent - Stateless rule engine for technical and event-driven signals.

Generates trading signals based on price movements, volume patterns,
moving averages, news events, and regulatory filings.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ========== DATA MODELS ==========

class SignalType(Enum):
    """Types of signals generated."""
    # Technical signals
    PRICE_DROP_1H = "price_drop_1h"
    PRICE_SPIKE_1H = "price_spike_1h"
    PRICE_DROP_24H = "price_drop_24h"
    PRICE_SPIKE_24H = "price_spike_24h"
    UNUSUAL_VOLUME = "unusual_volume"
    BEARISH_MA_CROSS = "bearish_ma_cross"
    BULLISH_MA_CROSS = "bullish_ma_cross"
    
    # Event-driven signals
    EVENT_POSITIVE = "event_positive"
    EVENT_NEGATIVE = "event_negative"
    FILING_MATERIAL = "filing_material"
    NEWS_CLUSTER_POSITIVE = "news_cluster_positive"
    NEWS_CLUSTER_NEGATIVE = "news_cluster_negative"


@dataclass
class Signal:
    """Trading signal with severity and explanation."""
    ticker: str
    type: str  # SignalType value
    severity: float  # 0.0 to 1.0
    timestamp: str
    explanation: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ========== DEFAULT THRESHOLDS ==========

DEFAULT_THRESHOLDS = {
    'price_drop_1h_pct': 2.0,      # -2% or more
    'price_spike_1h_pct': 2.0,     # +2% or more
    'price_drop_24h_pct': 5.0,     # -5% or more
    'price_spike_24h_pct': 5.0,    # +5% or more
    'volume_spike_ratio': 3.0,     # 3x average volume
    'ma_cross_threshold': 0.02,    # 2% difference for MA cross
    'sentiment_threshold': 0.3,    # Sentiment > 0.3 or < -0.3
    'filing_impact_material': True # Only material filings
}


# ========== TECHNICAL SIGNAL RULES ==========

def check_price_drop_1h(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for significant 1-hour price drop."""
    pct_change = market_data.get('pct_change_1h')
    if pct_change is None:
        return None
    
    threshold = thresholds.get('price_drop_1h_pct', 2.0)
    
    if pct_change <= -threshold:
        severity = min(1.0, abs(pct_change) / (threshold * 3))  # Scale to 1.0
        
        return Signal(
            ticker=ticker,
            type=SignalType.PRICE_DROP_1H.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Price dropped {abs(pct_change):.2f}% in last hour (threshold: {threshold}%)",
            metadata={'pct_change': pct_change, 'threshold': threshold}
        )
    
    return None


def check_price_spike_1h(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for significant 1-hour price spike."""
    pct_change = market_data.get('pct_change_1h')
    if pct_change is None:
        return None
    
    threshold = thresholds.get('price_spike_1h_pct', 2.0)
    
    if pct_change >= threshold:
        severity = min(1.0, pct_change / (threshold * 3))
        
        return Signal(
            ticker=ticker,
            type=SignalType.PRICE_SPIKE_1H.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Price spiked {pct_change:.2f}% in last hour (threshold: {threshold}%)",
            metadata={'pct_change': pct_change, 'threshold': threshold}
        )
    
    return None


def check_price_drop_24h(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for significant 24-hour price drop."""
    pct_change = market_data.get('pct_change_24h')
    if pct_change is None:
        return None
    
    threshold = thresholds.get('price_drop_24h_pct', 5.0)
    
    if pct_change <= -threshold:
        severity = min(1.0, abs(pct_change) / (threshold * 2))
        
        return Signal(
            ticker=ticker,
            type=SignalType.PRICE_DROP_24H.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Price dropped {abs(pct_change):.2f}% in last 24 hours (threshold: {threshold}%)",
            metadata={'pct_change': pct_change, 'threshold': threshold}
        )
    
    return None


def check_price_spike_24h(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for significant 24-hour price spike."""
    pct_change = market_data.get('pct_change_24h')
    if pct_change is None:
        return None
    
    threshold = thresholds.get('price_spike_24h_pct', 5.0)
    
    if pct_change >= threshold:
        severity = min(1.0, pct_change / (threshold * 2))
        
        return Signal(
            ticker=ticker,
            type=SignalType.PRICE_SPIKE_24H.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Price spiked {pct_change:.2f}% in last 24 hours (threshold: {threshold}%)",
            metadata={'pct_change': pct_change, 'threshold': threshold}
        )
    
    return None


def check_unusual_volume(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for unusual trading volume."""
    volume = market_data.get('volume')
    # Note: We'd need historical average volume - using a placeholder
    # In production, you'd calculate 30-day average from historical data
    
    if volume is None:
        return None
    
    # Placeholder: assume we have avg_volume_30d in metadata
    avg_volume = market_data.get('avg_volume_30d', volume * 0.5)  # Placeholder
    
    if avg_volume == 0:
        return None
    
    ratio = volume / avg_volume
    threshold = thresholds.get('volume_spike_ratio', 3.0)
    
    if ratio > threshold:
        severity = min(1.0, (ratio - threshold) / (threshold * 2))
        
        return Signal(
            ticker=ticker,
            type=SignalType.UNUSUAL_VOLUME.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Volume {ratio:.1f}x average (threshold: {threshold}x)",
            metadata={'volume': volume, 'avg_volume': avg_volume, 'ratio': ratio}
        )
    
    return None


def check_ma_crossover(
    market_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> Optional[Signal]:
    """Check for moving average crossovers (50MA vs 200MA)."""
    # Note: Requires historical price data to calculate MAs
    # This is a placeholder implementation
    
    ma_50 = market_data.get('ma_50')
    ma_200 = market_data.get('ma_200')
    
    if ma_50 is None or ma_200 is None:
        return None
    
    threshold = thresholds.get('ma_cross_threshold', 0.02)
    diff_pct = (ma_50 - ma_200) / ma_200 if ma_200 != 0 else 0
    
    # Bearish cross: 50MA crosses below 200MA
    if diff_pct < -threshold:
        severity = min(1.0, abs(diff_pct) / (threshold * 5))
        
        return Signal(
            ticker=ticker,
            type=SignalType.BEARISH_MA_CROSS.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"50MA crossed below 200MA (diff: {diff_pct*100:.2f}%)",
            metadata={'ma_50': ma_50, 'ma_200': ma_200, 'diff_pct': diff_pct}
        )
    
    # Bullish cross: 50MA crosses above 200MA
    elif diff_pct > threshold:
        severity = min(1.0, diff_pct / (threshold * 5))
        
        return Signal(
            ticker=ticker,
            type=SignalType.BULLISH_MA_CROSS.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"50MA crossed above 200MA (diff: {diff_pct*100:.2f}%)",
            metadata={'ma_50': ma_50, 'ma_200': ma_200, 'diff_pct': diff_pct}
        )
    
    return None


# ========== EVENT-DRIVEN SIGNAL RULES ==========

def check_news_cluster_signals(
    news_clusters: List[Dict[str, Any]],
    sentiment_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> List[Signal]:
    """Check for news cluster-based signals."""
    signals = []
    
    if not news_clusters or not sentiment_data:
        return signals
    
    # Get cluster sentiment
    cluster_sentiment = sentiment_data.get('cluster_sentiment', {})
    sentiment_threshold = thresholds.get('sentiment_threshold', 0.3)
    
    for cluster in news_clusters:
        cluster_id = cluster.get('id')
        if not cluster_id:
            continue
        
        cluster_sent = cluster_sentiment.get(cluster_id, {})
        sent_score = cluster_sent.get('sentiment_score', 0.0)
        
        # Positive news cluster
        if sent_score >= sentiment_threshold:
            severity = min(1.0, sent_score / 1.0)
            
            signals.append(Signal(
                ticker=ticker,
                type=SignalType.NEWS_CLUSTER_POSITIVE.value,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                explanation=f"Positive news cluster: {cluster.get('representative_headline', 'N/A')[:50]}...",
                metadata={'cluster_id': cluster_id, 'sentiment': sent_score}
            ))
        
        # Negative news cluster
        elif sent_score <= -sentiment_threshold:
            severity = min(1.0, abs(sent_score) / 1.0)
            
            signals.append(Signal(
                ticker=ticker,
                type=SignalType.NEWS_CLUSTER_NEGATIVE.value,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                explanation=f"Negative news cluster: {cluster.get('representative_headline', 'N/A')[:50]}...",
                metadata={'cluster_id': cluster_id, 'sentiment': sent_score}
            ))
    
    return signals


def check_filing_signals(
    filings: List[Dict[str, Any]],
    ticker: str,
    thresholds: Dict[str, Any]
) -> List[Signal]:
    """Check for filing-based signals."""
    signals = []
    
    if not filings:
        return signals
    
    ticker_filings = [f for f in filings if f.get('ticker') == ticker]
    
    for filing in ticker_filings:
        impact_tag = filing.get('impact_tag', 'immaterial')
        
        # Only material filings if threshold set
        if thresholds.get('filing_impact_material', True) and impact_tag != 'material':
            continue
        
        # Map impact to severity
        severity_map = {'material': 0.8, 'moderate': 0.5, 'immaterial': 0.2}
        severity = severity_map.get(impact_tag, 0.2)
        
        signals.append(Signal(
            ticker=ticker,
            type=SignalType.FILING_MATERIAL.value,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            explanation=f"Material filing: {filing.get('type', 'N/A')} - {filing.get('short_summary', 'N/A')[:50]}...",
            metadata={'filing_id': filing.get('id'), 'impact': impact_tag, 'type': filing.get('type')}
        ))
    
    return signals


def check_combined_event_signals(
    filings: List[Dict[str, Any]],
    sentiment_data: Dict[str, Any],
    ticker: str,
    thresholds: Dict[str, float]
) -> List[Signal]:
    """Check for combined event signals (filing + sentiment)."""
    signals = []
    
    if not filings or not sentiment_data:
        return signals
    
    ticker_filings = [f for f in filings if f.get('ticker') == ticker]
    ticker_sentiment = sentiment_data.get('ticker_sentiment', {}).get(ticker, {})
    sent_score = ticker_sentiment.get('sentiment_score', 0.0)
    
    for filing in ticker_filings:
        impact_tag = filing.get('impact_tag', 'immaterial')
        
        if impact_tag != 'material':
            continue
        
        sentiment_threshold = thresholds.get('sentiment_threshold', 0.3)
        
        # Material filing + negative sentiment = severe negative event
        if sent_score <= -sentiment_threshold:
            severity = min(1.0, abs(sent_score) * 1.2)  # Amplify severity
            
            signals.append(Signal(
                ticker=ticker,
                type=SignalType.EVENT_NEGATIVE.value,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                explanation=f"Material filing with negative sentiment: {filing.get('type', 'N/A')}",
                metadata={'filing_id': filing.get('id'), 'sentiment': sent_score, 'impact': impact_tag}
            ))
        
        # Material filing + positive sentiment = positive event
        elif sent_score >= sentiment_threshold:
            severity = min(1.0, sent_score * 1.2)
            
            signals.append(Signal(
                ticker=ticker,
                type=SignalType.EVENT_POSITIVE.value,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                explanation=f"Material filing with positive sentiment: {filing.get('type', 'N/A')}",
                metadata={'filing_id': filing.get('id'), 'sentiment': sent_score, 'impact': impact_tag}
            ))
    
    return signals


# ========== MAIN AGENT ==========

class SignalAgent:
    """
    Stateless rule engine for generating trading signals.
    """
    
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """
        Initialize SignalAgent.
        
        Args:
            thresholds: Custom thresholds for signal rules (optional)
        """
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    
    def run(
        self,
        market_data: List[Dict[str, Any]],
        news_clusters: Optional[List[Dict[str, Any]]] = None,
        filings: Optional[List[Dict[str, Any]]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate signals for all tickers.
        
        Args:
            market_data: List of market data per ticker
            news_clusters: Optional news clusters
            filings: Optional regulatory filings
            sentiment_data: Optional sentiment analysis results
            
        Returns:
            Dict with 'signals' list and summary stats
        """
        all_signals = []
        
        # Process each ticker
        for md in market_data:
            ticker = md.get('ticker')
            if not ticker:
                continue
            
            ticker_signals = []
            
            # Technical signals
            signal = check_price_drop_1h(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            signal = check_price_spike_1h(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            signal = check_price_drop_24h(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            signal = check_price_spike_24h(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            signal = check_unusual_volume(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            signal = check_ma_crossover(md, ticker, self.thresholds)
            if signal:
                ticker_signals.append(signal)
            
            # Event-driven signals
            if news_clusters and sentiment_data:
                signals = check_news_cluster_signals(news_clusters, sentiment_data, ticker, self.thresholds)
                ticker_signals.extend(signals)
            
            if filings:
                signals = check_filing_signals(filings, ticker, self.thresholds)
                ticker_signals.extend(signals)
            
            if filings and sentiment_data:
                signals = check_combined_event_signals(filings, sentiment_data, ticker, self.thresholds)
                ticker_signals.extend(signals)
            
            all_signals.extend(ticker_signals)
        
        # Calculate summary stats
        signal_counts = {}
        for signal in all_signals:
            sig_type = signal.type
            signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
        
        avg_severity = sum(s.severity for s in all_signals) / len(all_signals) if all_signals else 0.0
        
        return {
            'signals': [s.to_dict() for s in all_signals],
            'total_signals': len(all_signals),
            'signal_counts': signal_counts,
            'avg_severity': avg_severity,
            'timestamp': datetime.now().isoformat()
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    import json
    
    # Example data
    market_data = [
        {
            'ticker': 'AAPL',
            'price': 150.0,
            'pct_change_1h': -2.5,  # Triggers price_drop_1h
            'pct_change_24h': 3.0,
            'pct_change_7d': 5.0,
            'volume': 100000000,
            'avg_volume_30d': 25000000  # 4x average - triggers unusual_volume
        },
        {
            'ticker': 'GOOGL',
            'price': 140.0,
            'pct_change_1h': 0.5,
            'pct_change_24h': -6.0,  # Triggers price_drop_24h
            'pct_change_7d': -2.0,
            'volume': 50000000
        }
    ]
    
    sentiment_data = {
        'ticker_sentiment': {
            'AAPL': {'sentiment_score': 0.45},
            'GOOGL': {'sentiment_score': -0.35}
        },
        'cluster_sentiment': {
            'earnings_1': {'sentiment_score': 0.6}
        }
    }
    
    news_clusters = [
        {'id': 'earnings_1', 'representative_headline': 'Apple reports strong earnings'}
    ]
    
    filings = [
        {
            'id': 'f1',
            'ticker': 'AAPL',
            'type': '10-Q',
            'impact_tag': 'material',
            'short_summary': 'Quarterly earnings report'
        }
    ]
    
    # Run agent
    agent = SignalAgent()
    results = agent.run(
        market_data=market_data,
        news_clusters=news_clusters,
        filings=filings,
        sentiment_data=sentiment_data
    )
    
    # Display results
    print("="*70)
    print("SIGNALS GENERATED")
    print("="*70)
    print(json.dumps(results, indent=2))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Signals: {results['total_signals']}")
    print(f"Average Severity: {results['avg_severity']:.2f}")
    print("\nSignal Breakdown:")
    for sig_type, count in results['signal_counts'].items():
        print(f"  {sig_type}: {count}")
    
    print("\n" + "="*70)
    print("SIGNAL DETAILS")
    print("="*70)
    for signal in results['signals']:
        print(f"\n{signal['ticker']}: {signal['type'].upper()}")
        print(f"  Severity: {signal['severity']:.2f}")
        print(f"  {signal['explanation']}")
