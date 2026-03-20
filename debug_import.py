#!/usr/bin/env python3
import requests
import time

def test_reddit_direct():
    headers = {
        'User-Agent': 'TradingBot/1.0 (market sentiment aggregator - contact: kumardahal536@gmail.com)'
    }
    
    print("=" * 60)
    print("🔍 TESTING REDDIT API DIRECTLY")
    print("=" * 60)
    
    test_urls = [
        ("Hot posts - wallstreetbets", "https://www.reddit.com/r/wallstreetbets/hot.json?limit=5"),
        ("Hot posts - stocks", "https://www.reddit.com/r/stocks/hot.json?limit=5"),
        ("Search AAPL - wallstreetbets", "https://www.reddit.com/r/wallstreetbets/search.json?q=AAPL&restrict_sr=1&sort=new&limit=5"),
        ("Search AAPL - stocks", "https://www.reddit.com/r/stocks/search.json?q=AAPL&restrict_sr=1&sort=new&limit=5"),
        ("Search TSLA - wallstreetbets", "https://www.reddit.com/r/wallstreetbets/search.json?q=TSLA&restrict_sr=1&sort=new&limit=5"),
    ]
    
    for name, url in test_urls:
        print(f"\n📌 {name}")
        print(f"URL: {url}")
        print("-" * 40)
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'children' in data['data']:
                    posts = data['data']['children']
                    print(f"Found {len(posts)} posts")
                    
                    for i, post in enumerate(posts[:3]):
                        post_data = post['data']
                        title = post_data.get('title', 'No title')
                        print(f"\n  [{i+1}] {title[:100]}")
                        print(f"       Score: {post_data.get('score', 0)}")
                        print(f"       Subreddit: {post_data.get('subreddit', 'unknown')}")
                        print(f"       URL: https://reddit.com{post_data.get('permalink', '')}")
                else:
                    print(f"Unexpected response structure: {list(data.keys())}")
            else:
                print(f"Error: {response.text[:200]}")
                
                if response.status_code == 429:
                    print("⚠️ RATE LIMITED! Wait 5-10 minutes")
                elif response.status_code == 403:
                    print("⚠️ FORBIDDEN! User-Agent might be blocked")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()
        time.sleep(2)
    
    print("=" * 60)

if __name__ == "__main__":
    test_reddit_direct()


