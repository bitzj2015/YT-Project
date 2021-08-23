from youtube_transcript_api import YouTubeTranscriptApi
proxies = {
    'http': 'http://68.208.51.61:8080',
    'https': 'https://68.208.51.61:8080'
}

YouTubeTranscriptApi.get_transcript("hWKeJBYD0Dg",proxies=proxies)