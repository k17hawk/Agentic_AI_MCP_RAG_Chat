# 🤖 Agentic AI Trading System

> A fully autonomous, production-grade algorithmic trading system powered by multi-agent AI, MCP (Model Context Protocol), and RAG (Retrieval-Augmented Generation).

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](docker/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Configured-blue?logo=kubernetes)](kubernetes/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Modules](#-modules)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌐 Overview

The **Agentic AI Trading System** is an end-to-end autonomous trading platform that combines:

- 🧠 **Agentic AI** — Multi-step reasoning and decision-making pipelines
- 🔗 **MCP (Model Context Protocol)** — Structured tool use and memory coordination
- 📚 **RAG (Retrieval-Augmented Generation)** — Real-time knowledge from news, filings, and market data
- 🛡️ **Human-in-the-Loop (HITL)** — Configurable human approval gates before execution
- 📈 **Continuous Learning** — Bayesian weight updates, RL-based exit optimization, and genetic parameter tuning

---

## 🏗️ Architecture

```
Market Triggers → Discovery → Prefilter → Analysis → Risk → Portfolio → HITL → Execution
                                                                              ↑
                                                                         Memory & Learning
```

The system runs continuously, reacting to market events via a trigger layer and routing candidates through a progressive funnel of analysis, risk gating, and human approval before any order touches a broker.

---

## 📁 Project Structure

```
agentic_trading_system/
│
├── 📁 config/                          # Global configuration
│   ├── settings.py                     # Base settings & environment variables
│   ├── triggers.yaml                   # Trigger thresholds & schedules
│   ├── analysis_weights.yaml           # Technical/fundamental/sentiment weights by regime
│   ├── risk_config.yaml                # Risk parameters by asset class
│   ├── logging_config.yaml             # Logging configuration
│   └── database.yaml                   # DB connection settings
│
├── 📁 orchestrator/                    # System entry point & lifecycle management
│   ├── main.py                         # Starts the continuous system
│   ├── scheduler.py                    # APScheduler/Cron job manager
│   ├── state_manager.py                # Central state (Redis/Postgres)
│   ├── circuit_breaker.py              # API rate limiting & error handling
│   ├── health_check.py                 # System health monitoring
│   ├── graceful_shutdown.py            # SIGTERM handling
│   └── recovery.py                     # Crash recovery mechanisms
│
├── 📁 triggers/                        # Event-driven signal detection
│   ├── base_trigger.py                 # Abstract trigger class
│   ├── trigger_orchestrator.py         # Coordinates all triggers
│   ├── trigger_fusion.py               # Combines multiple trigger signals
│   ├── scheduled_trigger.py            # Time-based triggers
│   ├── price_alert_trigger.py          # Multi-timeframe price movement
│   │   ├── sliding_window.py           # Rolling window calculations
│   │   ├── volatility_adjusted.py      # Dynamic thresholds
│   │   └── statistical_significance.py # Z-score, t-test detection
│   ├── news_alert_trigger.py           # News sentiment triggers
│   │   ├── news_api_client.py          # NewsAPI, Alpha Vantage, etc.
│   │   └── sentiment_scorer.py         # NLP sentiment analysis
│   ├── volume_spike_trigger.py         # Unusual volume detection
│   ├── pattern_recognition_trigger.py  # Chart pattern detection
│   │   ├── candlestick_patterns.py     # Doji, engulfing, hammer, etc.
│   │   └── technical_patterns.py       # Head & shoulders, double top
│   └── social_sentiment_trigger.py     # Twitter/Reddit monitoring
│       ├── twitter_client.py           # Twitter API v2
│       └── reddit_client.py            # PRAW client
│
├── 📁 discovery/                       # Data aggregation & entity extraction
│   ├── search_aggregator.py            # Coordinates all search sources
│   ├── tavily_client.py                # Tavily API wrapper
│   ├── news_api_client.py              # Multiple news sources
│   ├── social_media_client.py          # Twitter/Reddit
│   ├── sec_filings_client.py           # EDGAR API for insider trades
│   ├── options_flow_client.py          # Unusual options activity
│   ├── macro_data_client.py            # Economic indicators
│   ├── entity_extractor.py             # Extract tickers/companies
│   │   ├── nlp_extractor.py            # Spacy/NER
│   │   └── regex_extractor.py          # Pattern matching
│   └── data_enricher.py                # Enrich with additional context
│
├── 📁 prefilter/                       # Quality gating before deep analysis
│   ├── quality_gates.py                # Main filtering orchestrator
│   ├── exchange_validator.py           # Check allowed exchanges
│   ├── price_range_checker.py          # Min/max price validation
│   ├── volume_checker.py               # Liquidity requirements
│   ├── market_cap_checker.py           # Size requirements
│   ├── data_quality_checker.py         # Sufficient history check
│   ├── rejected_logger.py              # Store rejection reasons
│   └── passed_queue.py                 # Queue for analysis
│
├── 📁 analysis/                        # Multi-dimensional signal analysis
│   ├── analysis_orchestrator.py        # Coordinates all analysis modules
│   ├── multi_timeframe_aggregator.py   # Combines signals across timeframes
│   ├── regime_detector.py              # Market regime classification
│   │   ├── volatility_regime.py        # VIX-based regime
│   │   ├── trend_regime.py             # ADX, moving averages
│   │   └── correlation_regime.py       # Sector correlation
│   │
│   ├── 📁 technical/                   # Technical analysis engine
│   │   ├── technical_analyzer.py       # Main technical analysis
│   │   ├── indicators/
│   │   │   ├── trend.py                # MA, EMA, MACD, Ichimoku
│   │   │   ├── momentum.py             # RSI, Stochastic, Williams %R
│   │   │   ├── volume.py               # OBV, MFI, VWAP
│   │   │   ├── volatility.py           # Bollinger, Keltner, ATR
│   │   │   └── custom.py               # Composite indicators
│   │   ├── patterns/
│   │   │   ├── candlestick.py          # Pattern recognition
│   │   │   ├── chart_patterns.py       # Support/resistance
│   │   │   └── harmonic.py             # Harmonic patterns
│   │   ├── timeframe_analysis.py       # Multi-timeframe analysis
│   │   │   ├── intraday.py
│   │   │   ├── daily.py
│   │   │   ├── weekly.py
│   │   │   └── monthly.py
│   │   └── technical_scorer.py         # Score calculation
│   │
│   ├── 📁 fundamental/                 # Fundamental analysis engine
│   │   ├── fundamental_analyzer.py     # Main fundamental analysis
│   │   ├── valuation.py                # P/E, P/B, P/S, EV/EBITDA
│   │   ├── growth.py                   # Revenue/EPS growth
│   │   ├── profitability.py            # ROE, ROA, margins
│   │   ├── liquidity.py                # Current/quick ratio
│   │   ├── solvency.py                 # D/E, interest coverage
│   │   ├── efficiency.py               # Asset turnover
│   │   ├── discounted_cash_flow.py     # DCF valuation
│   │   └── fundamental_scorer.py       # Score calculation
│   │
│   ├── 📁 sentiment/                   # Sentiment analysis engine
│   │   ├── sentiment_analyzer.py       # Main sentiment analysis
│   │   ├── news_sentiment.py           # News articles
│   │   ├── social_sentiment.py         # Social media
│   │   ├── analyst_ratings.py          # Analyst consensus
│   │   ├── insider_activity.py         # Insider transactions
│   │   ├── institutional_holdings.py   # 13F filings
│   │   └── sentiment_scorer.py         # Score calculation
│   │
│   └── weighted_score_engine.py        # Combines all scores with dynamic weights
│
├── 📁 risk/                            # Risk management & position sizing
│   ├── risk_manager.py                 # Main risk orchestrator
│   ├── market_regime_risk.py           # Regime-based adjustments
│   ├── position_sizing/
│   │   ├── kelly_criterion.py          # Kelly formula
│   │   ├── half_kelly.py               # Conservative Kelly
│   │   ├── fixed_fraction.py           # Fixed % risk
│   │   └── volatility_adjusted.py      # ATR-based sizing
│   ├── stop_loss_optimizer.py          # Dynamic stop placement
│   │   ├── atr_stop.py                 # ATR-based stops
│   │   ├── volatility_stop.py          # Volatility-adjusted
│   │   ├── trailing_stop.py            # Trailing stops
│   │   └── time_stop.py                # Time-based exits
│   ├── portfolio_risk/
│   │   ├── var_calculator.py           # Value at Risk
│   │   ├── expected_shortfall.py       # CVaR
│   │   ├── correlation_matrix.py       # Portfolio correlation
│   │   ├── diversification_score.py    # Sector exposure
│   │   └── stress_tester.py            # Monte Carlo stress testing
│   ├── risk_scorer.py                  # Pass/Fail decision
│   └── risk_approved_queue.py          # Approved stocks queue
│
├── 📁 portfolio/                       # Portfolio optimization & allocation
│   ├── portfolio_optimizer.py          # Main optimizer
│   ├── efficient_frontier.py           # Markowitz model
│   ├── black_litterman.py              # Black-Litterman model
│   ├── risk_parity.py                  # Risk parity allocation
│   ├── hierarchical_risk_parity.py     # HRP
│   ├── allocation_engine.py            # Final weight calculation
│   ├── rebalancing_signals.py          # Rebalance triggers
│   └── recommendation_generator.py    # BUY / SELL / HOLD signals
│
├── 📁 hitl/                            # Human-in-the-Loop approval layer
│   ├── alert_manager.py                # Coordinates all alerts
│   ├── channels/
│   │   ├── whatsapp_client.py          # Twilio WhatsApp
│   │   ├── email_client.py             # SMTP / SendGrid
│   │   ├── sms_client.py               # Twilio SMS
│   │   └── dashboard_notifier.py       # Web dashboard push
│   ├── message_builder.py              # Format alert messages
│   ├── response_parser.py              # Parse human replies
│   ├── pending_queue.py                # Awaiting human response
│   ├── timeout_manager.py              # Auto-reject on timeout
│   ├── decision_tracker.py             # Store human decisions
│   └── feedback_logger.py              # Feed decisions back to discovery
│
├── 📁 execution/                       # Order management & broker connectivity
│   ├── execution_engine.py             # Main execution orchestrator
│   ├── order_manager.py                # Order lifecycle management
│   ├── order_types/
│   │   ├── market_order.py
│   │   ├── limit_order.py
│   │   ├── stop_order.py
│   │   └── trailing_stop_order.py
│   ├── broker_connectors/
│   │   ├── alpaca_client.py            # Alpaca API
│   │   ├── ibkr_client.py              # Interactive Brokers
│   │   ├── paper_trading.py            # Simulation mode
│   │   └── mock_broker.py              # Unit testing
│   ├── routing/
│   │   ├── smart_order_routing.py      # Best execution routing
│   │   └── venue_analyzer.py           # Liquidity analysis
│   ├── fills_manager.py                # Track executions
│   ├── open_positions.py               # Current holdings
│   └── settlement.py                   # Cash management
│
├── 📁 memory/                          # Tiered memory & persistence layer
│   ├── memory_orchestrator.py          # Coordinates all memory tiers
│   ├── models.py                       # Pydantic/SQLAlchemy models
│   ├── repositories/
│   │   ├── trade_repository.py         # Trade CRUD
│   │   ├── signal_repository.py        # Signal history
│   │   ├── performance_repository.py   # Performance metrics
│   │   └── model_weights_repository.py # ML model weights
│   ├── short_term/
│   │   ├── redis_client.py             # Redis connection
│   │   └── session_cache.py            # Current session cache
│   ├── medium_term/
│   │   ├── postgres_client.py          # PostgreSQL
│   │   └── warehouse.py                # 90-day rolling storage
│   ├── long_term/
│   │   ├── s3_client.py                # AWS S3 / MinIO
│   │   ├── data_lake.py                # Parquet/Feather storage
│   │   └── archive_manager.py          # Cold storage management
│   └── query_engine.py                 # Unified memory query API
│
├── 📁 learning/                        # Continuous learning & adaptation
│   ├── learning_orchestrator.py        # Main learning coordinator
│   ├── feature_store.py                # Feature engineering
│   ├── attribution_engine.py           # Signal attribution analysis
│   ├── models/
│   │   ├── weight_optimizer.py         # Bayesian weight updating
│   │   ├── genetic_algorithm.py        # Parameter tuning
│   │   ├── reinforcement_learning.py   # RL-based exit optimization
│   │   └── ensemble_model.py           # Model stacking
│   ├── backtester.py                   # Historical validation
│   │   ├── simulation_engine.py
│   │   ├── monte_carlo.py
│   │   └── walk_forward.py
│   ├── forward_tester.py               # Paper trading validation
│   └── config_updater.py               # Auto-update YAML configs
│
├── 📁 analytics/                       # Performance analytics & dashboards
│   ├── metrics_engine.py               # Main metrics calculator
│   ├── performance_metrics/
│   │   ├── pnl_calculator.py           # Profit/Loss
│   │   ├── sharpe_ratio.py
│   │   ├── sortino_ratio.py
│   │   ├── calmar_ratio.py
│   │   ├── win_rate.py
│   │   ├── profit_factor.py
│   │   ├── max_drawdown.py
│   │   └── recovery_factor.py
│   ├── attribution/
│   │   ├── signal_attribution.py       # Signal contribution analysis
│   │   ├── factor_attribution.py       # Factor model attribution
│   │   └── alpha_decay.py              # Signal half-life analysis
│   ├── dashboards/
│   │   ├── plot_generator.py           # Matplotlib/Plotly charts
│   │   ├── html_reporter.py            # Interactive HTML dashboard
│   │   └── pdf_generator.py            # PDF report generation
│   └── alerts_generator.py             # Performance-based alerts
│
├── 📁 reporting/                       # Report generation & distribution
│   ├── report_generator.py             # Main report builder
│   ├── templates/
│   │   ├── daily_digest.html
│   │   ├── weekly_report.html
│   │   ├── monthly_report.html
│   │   └── trade_confirmation.html
│   ├── pdf_builder.py                  # PDF generation
│   ├── email_builder.py                # HTML email formatting
│   ├── whatsapp_builder.py             # WhatsApp message formatting
│   ├── export_engine.py                # CSV/JSON export
│   ├── compliance_logger.py            # Full audit trail
│   └── archive_manager.py              # Report archival
│
├── 📁 utils/                           # Shared utilities & helpers
│   ├── decorators.py                   # Logging, retry, timing decorators
│   ├── helpers.py                      # General utility functions
│   ├── validators.py                   # Input validation
│   ├── exceptions.py                   # Custom exception classes
│   ├── constants.py                    # System-wide constants
│   ├── date_utils.py                   # Date/time helpers
│   ├── number_utils.py                 # Financial math utilities
│   └── singleton.py                    # Singleton pattern helper
│
├── 📁 tests/                           # Full test suite
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/
│   │   ├── test_triggers/
│   │   ├── test_analysis/
│   │   ├── test_risk/
│   │   └── test_execution/
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_broker_connection.py
│   │   └── test_database.py
│   ├── performance/
│   │   ├── test_latency.py
│   │   └── test_throughput.py
│   └── mocks/
│       ├── mock_broker.py
│       ├── mock_yahoo.py
│       └── mock_news_api.py
│
├── 📁 data/                            # Local data storage
│   ├── raw/                            # Raw downloaded market data
│   ├── processed/                      # Cleaned & normalized data
│   ├── models/                         # Trained ML model artifacts
│   ├── reports/                        # Generated reports
│   ├── charts/                         # Generated chart images
│   └── logs/                           # Application log files
│
├── 📁 scripts/                         # DevOps & maintenance scripts
│   ├── setup_db.sh                     # Initialize databases
│   ├── run_migrations.py               # Alembic migrations
│   ├── seed_data.py                    # Seed test data
│   ├── backup.sh                       # Backup scripts
│   └── deploy.sh                       # Deployment automation
│
├── 📁 docker/                          # Container configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── .dockerignore
│
├── 📁 kubernetes/                      # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   └── ingress.yaml
│
├── .env.example                        # Environment variables template
├── .gitignore
├── pyproject.toml                      # Poetry/PDM dependency management
├── poetry.lock
├── requirements.txt                    # Pip requirements
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── Makefile                            # Common developer commands
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Redis
- PostgreSQL
- AWS S3 or MinIO (for long-term storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-trading-system.git
cd agentic-trading-system

# Copy environment variables
cp .env.example .env

# Install dependencies
pip install -r requirements.txt
# or with Poetry
poetry install

# Initialize the database
bash scripts/setup_db.sh
python scripts/run_migrations.py

# Seed test data (optional)
python scripts/seed_data.py
```

### Running with Docker

```bash
# Development
docker compose -f docker/docker-compose.dev.yml up

# Production
docker compose -f docker/docker-compose.yml up -d
```

### Running Locally

```bash
# Start the orchestrator
python orchestrator/main.py
```

---

## ⚙️ Configuration

All configuration lives in the `config/` directory:

| File | Purpose |
|------|---------|
| `settings.py` | Environment variables, API keys, broker credentials |
| `triggers.yaml` | Price movement thresholds, schedule intervals |
| `analysis_weights.yaml` | Per-regime weights for technical/fundamental/sentiment |
| `risk_config.yaml` | Max drawdown, position limits, VaR thresholds |
| `logging_config.yaml` | Log levels, handlers, rotation settings |
| `database.yaml` | Redis, PostgreSQL, S3 connection strings |

Copy `.env.example` to `.env` and fill in your API keys before running.

---

## 🧩 Modules

| Module | Responsibility |
|--------|---------------|
| **Triggers** | Detect market events (price moves, news, volume spikes, social sentiment) |
| **Discovery** | Aggregate data from news APIs, SEC filings, options flow, social media |
| **Prefilter** | Gate candidates by exchange, price, volume, market cap, data quality |
| **Analysis** | Technical, fundamental, and sentiment scoring with regime-aware weights |
| **Risk** | Position sizing (Kelly, ATR), stop-loss optimization, portfolio VaR |
| **Portfolio** | Markowitz, Black-Litterman, HRP optimization; BUY/SELL/HOLD signals |
| **HITL** | WhatsApp/email/SMS alerts with human approval gates before execution |
| **Execution** | Order routing to Alpaca, IBKR, or paper trading simulation |
| **Memory** | Redis (hot), PostgreSQL (warm), S3/data lake (cold) tiered storage |
| **Learning** | Bayesian weight updates, RL exits, genetic parameter optimization |
| **Analytics** | Sharpe, Sortino, Calmar, win rate, drawdown, signal attribution |
| **Reporting** | Daily/weekly/monthly PDF & HTML reports, trade confirmations |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance benchmarks
pytest tests/performance/

# With coverage report
pytest --cov=. --cov-report=html
```

---

## 🚢 Deployment

### Kubernetes

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### Using Make

```bash
make install       # Install dependencies
make test          # Run test suite
make lint          # Run linting
make docker-build  # Build Docker image
make deploy        # Deploy to Kubernetes
make backup        # Run backup scripts
```

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting pull requests, code style, and the development workflow.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Trading involves substantial risk of loss. Always consult a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred through the use of this software.
