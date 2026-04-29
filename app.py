@@
-agent = DataScienceAgent()
+agent = DataScienceAgent()
+
+# Register API helper routes (forecast, ab_test, hf_status)
+try:
+    from core.api_helpers import api_bp
+    app.register_blueprint(api_bp)
+    # make agent accessible to the blueprint
+    app.config['agent'] = agent
+except Exception:
+    # if import fails, continue without blocking app startup
+    logger.info('Optional: core.api_helpers not available')
@@
