#!/bin/bash

# Define the root directory
ROOT="agentic_trading_system"

echo "🚀 Starting folder structure generation for: $ROOT"

# Create all directories in one go to ensure hierarchy
mkdir -p $ROOT/{config,orchestrator,triggers/price_alert_trigger,triggers/news_alert_trigger,triggers/social_sentiment_trigger,triggers/pattern_recognition_trigger,discovery/entity_extractor,prefilter,analysis/{technical/{indicators,patterns,timeframe_analysis},fundamental,sentiment},risk/{position_sizing,stop_loss_optimizer,portfolio_risk},portfolio,hitl/channels,execution/{order_types,broker_connectors,routing},memory/{repositories,short_term,medium_term,long_term},learning/models,analytics/{performance_metrics,attribution,dashboards},reporting/templates,utils,tests/{unit/{test_triggers,test_analysis,test_risk,test_execution},integration,performance,mocks},data/{raw,processed,models,reports,charts,logs},scripts,docker,kubernetes}

# --- 1. CONFIG ---
touch $ROOT/config/{__init__.py,settings.py,triggers.yaml,analysis_weights.yaml,risk_config.yaml,logging_config.yaml,database.yaml}

# --- 2. ORCHESTRATOR ---
touch $ROOT/orchestrator/{__init__.py,main.py,scheduler.py,state_manager.py,circuit_breaker.py,health_check.py,graceful_shutdown.py,recovery.py}

# --- 3. TRIGGERS ---
touch $ROOT/triggers/{__init__.py,base_trigger.py,trigger_orchestrator.py,trigger_fusion.py,scheduled_trigger.py,volume_spike_trigger.py}
touch $ROOT/triggers/price_alert_trigger/{sliding_window.py,volatility_adjusted.py,statistical_significance.py}
touch $ROOT/triggers/news_alert_trigger/{news_api_client.py,sentiment_scorer.py}
touch $ROOT/triggers/pattern_recognition_trigger/{candlestick_patterns.py,technical_patterns.py}
touch $ROOT/triggers/social_sentiment_trigger/{twitter_client.py,reddit_client.py}

# --- 4. DISCOVERY ---
touch $ROOT/discovery/{__init__.py,search_aggregator.py,tavily_client.py,news_api_client.py,social_media_client.py,sec_filings_client.py,options_flow_client.py,macro_data_client.py,data_enricher.py}
touch $ROOT/discovery/entity_extractor/{nlp_extractor.py,regex_extractor.py}

# --- 5. PREFILTER ---
touch $ROOT/prefilter/{__init__.py,quality_gates.py,exchange_validator.py,price_range_checker.py,volume_checker.py,market_cap_checker.py,data_quality_checker.py,rejected_logger.py,passed_queue.py}

# --- 6. ANALYSIS ---
touch $ROOT/analysis/{__init__.py,analysis_orchestrator.py,multi_timeframe_aggregator.py,regime_detector.py,weighted_score_engine.py}
touch $ROOT/analysis/technical/{__init__.py,technical_analyzer.py,technical_scorer.py}
touch $ROOT/analysis/technical/indicators/{trend.py,momentum.py,volume.py,volatility.py,custom.py}
touch $ROOT/analysis/technical/patterns/{candlestick.py,chart_patterns.py,harmonic.py}
touch $ROOT/analysis/technical/timeframe_analysis/{intraday.py,daily.py,weekly.py,monthly.py}
touch $ROOT/analysis/fundamental/{__init__.py,fundamental_analyzer.py,valuation.py,growth.py,profitability.py,liquidity.py,solvency.py,efficiency.py,discounted_cash_flow.py,fundamental_scorer.py}
touch $ROOT/analysis/sentiment/{__init__.py,sentiment_analyzer.py,news_sentiment.py,social_sentiment.py,analyst_ratings.py,insider_activity.py,institutional_holdings.py,sentiment_scorer.py}

# --- 7. RISK ---
touch $ROOT/risk/{__init__.py,risk_manager.py,market_regime_risk.py,risk_scorer.py,risk_approved_queue.py}
touch $ROOT/risk/position_sizing/{kelly_criterion.py,half_kelly.py,fixed_fraction.py,volatility_adjusted.py}
touch $ROOT/risk/stop_loss_optimizer/{atr_stop.py,volatility_stop.py,trailing_stop.py,time_stop.py}
touch $ROOT/risk/portfolio_risk/{var_calculator.py,expected_shortfall.py,correlation_matrix.py,diversification_score.py,stress_tester.py}

# --- 8. PORTFOLIO ---
touch $ROOT/portfolio/{__init__.py,portfolio_optimizer.py,efficient_frontier.py,black_litterman.py,risk_parity.py,hierarchical_risk_parity.py,allocation_engine.py,rebalancing_signals.py,recommendation_generator.py}

# --- 9. HITL ---
touch $ROOT/hitl/{__init__.py,alert_manager.py,message_builder.py,response_parser.py,pending_queue.py,timeout_manager.py,decision_tracker.py,feedback_logger.py}
touch $ROOT/hitl/channels/{whatsapp_client.py,email_client.py,sms_client.py,dashboard_notifier.py}

# --- 10. EXECUTION ---
touch $ROOT/execution/{__init__.py,execution_engine.py,order_manager.py,fills_manager.py,open_positions.py,settlement.py}
touch $ROOT/execution/order_types/{market_order.py,limit_order.py,stop_order.py,trailing_stop_order.py}
touch $ROOT/execution/broker_connectors/{alpaca_client.py,ibkr_client.py,paper_trading.py,mock_broker.py}
touch $ROOT/execution/routing/{smart_order_routing.py,venue_analyzer.py}

# --- 11. MEMORY ---
touch $ROOT/memory/{__init__.py,memory_orchestrator.py,models.py,query_engine.py}
touch $ROOT/memory/repositories/{trade_repository.py,signal_repository.py,performance_repository.py,model_weights_repository.py}
touch $ROOT/memory/short_term/{redis_client.py,session_cache.py}
touch $ROOT/memory/medium_term/{postgres_client.py,warehouse.py}
touch $ROOT/memory/long_term/{s3_client.py,data_lake.py,archive_manager.py}

# --- 12. LEARNING ---
touch $ROOT/learning/{__init__.py,learning_orchestrator.py,feature_store.py,attribution_engine.py,backtester.py,forward_tester.py,config_updater.py}
touch $ROOT/learning/models/{weight_optimizer.py,genetic_algorithm.py,reinforcement_learning.py,ensemble_model.py}

# --- 13. ANALYTICS ---
touch $ROOT/analytics/{__init__.py,metrics_engine.py,alerts_generator.py}
touch $ROOT/analytics/performance_metrics/{pnl_calculator.py,sharpe_ratio.py,sortino_ratio.py,calmar_ratio.py,win_rate.py,profit_factor.py,max_drawdown.py,recovery_factor.py}
touch $ROOT/analytics/attribution/{signal_attribution.py,factor_attribution.py,alpha_decay.py}
touch $ROOT/analytics/dashboards/{plot_generator.py,html_reporter.py,pdf_generator.py}

# --- 14. REPORTING ---
touch $ROOT/reporting/{__init__.py,report_generator.py,pdf_builder.py,email_builder.py,whatsapp_builder.py,export_engine.py,compliance_logger.py,archive_manager.py}
touch $ROOT/reporting/templates/{daily_digest.html,weekly_report.html,monthly_report.html,trade_confirmation.html}

# --- 15. UTILS & TESTS ---
touch $ROOT/utils/{__init__.py,decorators.py,helpers.py,validators.py,exceptions.py,constants.py,date_utils.py,number_utils.py,singleton.py}
touch $ROOT/tests/{__init__.py,conftest.py}
touch $ROOT/tests/integration/{test_full_pipeline.py,test_broker_connection.py,test_database.py}
touch $ROOT/tests/performance/{test_latency.py,test_throughput.py}
touch $ROOT/tests/mocks/{mock_broker.py,mock_yahoo.py,mock_news_api.py}

# --- 16. SCRIPTS, DOCKER, K8S & ROOT FILES ---
touch $ROOT/scripts/{setup_db.sh,run_migrations.py,seed_data.py,backup.sh,deploy.sh}
touch $ROOT/docker/{Dockerfile,docker-compose.yml,docker-compose.dev.yml,.dockerignore}
touch $ROOT/kubernetes/{deployment.yaml,service.yaml,configmap.yaml,secrets.yaml,ingress.yaml}

touch $ROOT/{.env.example,.gitignore,pyproject.toml,poetry.lock,requirements.txt,README.md,CHANGELOG.md,CONTRIBUTING.md,LICENSE,Makefile}

# Ensure scripts are executable
chmod +x $ROOT/scripts/*.sh

echo "✅ Structure created successfully!"