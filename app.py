AttributeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/besttradingbei/app.py", line 13, in <module>
    df = get_data(symbol, timeframe)
File "/mount/src/besttradingbei/data.py", line 12, in get_data
    df.columns = [c.lower() for c in df.columns]
