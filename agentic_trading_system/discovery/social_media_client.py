#!/usr/bin/env python3
"""
Patched SocialMediaClient that actually searches Reddit
"""
import aiohttp
import asyncio
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SocialMediaClient:
    """Wrapper that fixes Reddit search"""
    
    def __init__(self, original_client):
        self.original = original_client
        self.config = original_client.config if hasattr(original_client, 'config') else {}
        
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """Search with proper Reddit search"""
        options = options or {}
        
        print(f"\n📱 PATCHED: Searching Reddit for '{query}'")
        
        # Only handle Reddit, delegate other platforms to original
        if options.get('platforms') and 'reddit' not in options['platforms']:
            return await self.original.search(query, options)
        
        # Do direct Reddit search
        reddit_results = await self._search_reddit_direct(query, options)
        
        # Get results from other platforms via original client
        other_platforms = []
        if options.get('platforms'):
            other_platforms = [p for p in options['platforms'] if p != 'reddit']
        
        other_results = {'items': [], 'metadata': {}}
        if other_platforms:
            other_options = options.copy()
            other_options['platforms'] = other_platforms
            other_results = await self.original.search(query, other_options)
        
        # Combine results
        all_items = reddit_results.get('items', []) + other_results.get('items', [])
        
        return {
            'items': all_items,
            'metadata': {
                'total_found': len(all_items),
                'unique_count': len(all_items),
                'platforms_used': ['reddit'] + other_platforms,
                'platform_results': {
                    'reddit': {
                        'status': 'success',
                        'count': len(reddit_results.get('items', []))
                    },
                    **other_results.get('metadata', {}).get('platform_results', {})
                }
            }
        }
    
    async def _search_reddit_direct(self, query: str, options: Dict) -> Dict[str, Any]:
        """Direct Reddit search using the search endpoint"""
        
        # Get config
        reddit_config = self.config.get('reddit', {})
        subreddits = reddit_config.get('subreddits', ['wallstreetbets', 'stocks'])
        limit = options.get('reddit_limit', reddit_config.get('limit', 25))
        sort = options.get('reddit_sort', 'relevance')
        time_filter = options.get('reddit_time_filter', 'week')
        
        headers = {
            'User-Agent': reddit_config.get('user_agent', 'TradingBot/1.0')
        }
        
        all_posts = []
        
        print(f"   🔍 Searching {len(subreddits)} subreddits for '{query}'")
        
        async with aiohttp.ClientSession() as session:
            for subreddit in subreddits:
                try:
                    # Use the search endpoint
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': query,
                        'restrict_sr': 'true',
                        'sort': sort,
                        't': time_filter,
                        'limit': limit
                    }
                    
                    print(f"   📌 r/{subreddit}: searching...")
                    
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'data' in data and 'children' in data['data']:
                                posts = data['data']['children']
                                print(f"      ✅ Found {len(posts)} posts")
                                
                                for post in posts:
                                    post_data = post['data']
                                    all_posts.append({
                                        'title': post_data.get('title', ''),
                                        'content': post_data.get('selftext', '')[:500],
                                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                        'source': 'reddit',
                                        'platform': 'reddit',
                                        'subreddit': subreddit,
                                        'score': post_data.get('score', 0),
                                        'num_comments': post_data.get('num_comments', 0),
                                        'created_utc': post_data.get('created_utc', 0),
                                        'relevance_score': 1.0  # All search results are relevant
                                    })
                            else:
                                print(f"      ⚠️ Unexpected response format")
                        elif response.status == 429:
                            print(f"      ⚠️ Rate limited on r/{subreddit}")
                            await asyncio.sleep(5)
                        else:
                            print(f"      ❌ Error {response.status}")
                            
                except Exception as e:
                    print(f"      ❌ Exception: {e}")
                
                # Small delay between subreddits
                await asyncio.sleep(1)
        
        print(f"   📊 Total posts found: {len(all_posts)}")
        
        return {
            'items': all_posts,
            'metadata': {
                'count': len(all_posts),
                'query': query
            }
        }