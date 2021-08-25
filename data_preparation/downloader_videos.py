#!/usr/bin/python3
'''
This script comes from https://github.com/egbertbouman/youtube-comment-downloader
'''
from __future__ import print_function

import argparse
import io
import json
import os
import sys
import time

import re
import requests
from requests.models import Response

YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={youtube_id}'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'

SORT_BY_POPULAR = 0
SORT_BY_RECENT = 1

YT_CFG_RE = r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;'
YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def ajax_request(session, endpoint, ytcfg, retries=5, sleep=20):
    url = 'https://www.youtube.com' + endpoint['commandMetadata']['webCommandMetadata']['apiUrl']
    
    data = {'context': ytcfg['INNERTUBE_CONTEXT'],
            'continuation': endpoint['continuationCommand']['token']}

    for _ in range(retries):
        response = session.post(url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data)
        if response.status_code == 200:
            return response.json()
        if response.status_code in [403, 413]:
            return {}
        else:
            time.sleep(sleep)


def download_comments(youtube_id, sort_by=SORT_BY_RECENT, sleep=.1, proxy=''):
    session = requests.Session()
    if proxy != '':
        proxies = {
            'https': proxy
        }
        session.proxies.update(proxies)
    session.headers['User-Agent'] = USER_AGENT

    response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))

    if 'uxe=' in response.request.url:
        session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
        response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))

    html = response.text
    ytcfg = json.loads(regex_search(html, YT_CFG_RE, default=''))
    
    if not ytcfg:
        return # Unable to extract configuration

    data = json.loads(regex_search(html, YT_INITIAL_DATA_RE, default=''))
    videoIds = {}

    for recvideo in reversed(list(search_dict(data, 'compactVideoRenderer'))):
        if recvideo['videoId'] not in videoIds.keys():
            videoIds[recvideo['videoId']] = 1
            yield {'videoId': recvideo['videoId'],
                    'title': recvideo['accessibility']['accessibilityData']['label']
            }

    section = next(search_dict(data, 'secondaryResults'), None)
    renderer = next(search_dict(section, 'continuationItemRenderer'), None) if section else None
    if not renderer:
        # Comments disabled?
        return

    continuations = [renderer['continuationEndpoint']]
    while continuations:
        continuation = continuations.pop()
        response = ajax_request(session, continuation, ytcfg)
        # print(response)
        if not response:
            break
        if list(search_dict(response, 'externalErrorMessage')):
            raise RuntimeError('Error returned from server: ' + next(search_dict(response, 'externalErrorMessage')))

        actions = list(search_dict(response, 'reloadContinuationItemsCommand')) + \
                  list(search_dict(response, 'appendContinuationItemsAction'))
        for action in actions:
            for item in action.get('continuationItems', []):
                continuations[:0] = [ep for ep in search_dict(item, 'continuationEndpoint')]
        
        for recvideo in reversed(list(search_dict(response, 'compactVideoRenderer'))):
            if recvideo['videoId'] not in videoIds.keys():
                videoIds[recvideo['videoId']] = 1
                yield {'videoId': recvideo['videoId'],
                        'title': recvideo['accessibility']['accessibilityData']['label']
                }
        time.sleep(sleep)


def search_dict(partial, search_key):
    stack = [partial]
    while stack:
        current_item = stack.pop()
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                if key == search_key:
                    yield value
                else:
                    stack.append(value)
        elif isinstance(current_item, list):
            for value in current_item:
                stack.append(value)


def main(argv = None):
    parser = argparse.ArgumentParser(add_help=False, description=('Download Youtube recvideos without using the Youtube API'))
    parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')
    parser.add_argument('--youtubeid', '-y', help='ID of Youtube video for which to download the recvideos')
    parser.add_argument('--output', '-o', help='Output filename (output format is line delimited JSON)')
    parser.add_argument('--limit', '-l', type=int, help='Limit the number of recvideos')
    parser.add_argument('--sort', '-s', type=int, default=SORT_BY_RECENT,
                        help='Whether to download popular (0) or recent recvideos (1). Defaults to 1')
    parser.add_argument('--sleep', '-st', type=float, default=0.1, help='sleep time')
    parser.add_argument('--proxy', '-p', type=str, default='', help='whether to use proxy url, e.g. http://68.208.51.61:8080')

    try:
        args = parser.parse_args() if argv is None else parser.parse_args(argv)

        youtube_id = args.youtubeid
        output = args.output
        limit = args.limit

        if not youtube_id or not output:
            parser.print_usage()
            raise ValueError('you need to specify a Youtube ID and an output filename')

        if os.sep in output:
            outdir = os.path.dirname(output)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        print('Downloading Youtube recvideos for video:', youtube_id)
        count = 0
        with io.open(output, 'w', encoding='utf8') as fp:
            sys.stdout.write('Downloaded %d recvideo(s)\r' % count)
            sys.stdout.flush()
            start_time = time.time()
            for recvideo in download_comments(youtube_id, args.sort, sleep=args.sleep, proxy=args.proxy):
                recvideo_json = json.dumps(recvideo, ensure_ascii=False)
                print(recvideo_json.decode('utf-8') if isinstance(recvideo_json, bytes) else recvideo_json, file=fp)
                count += 1
                sys.stdout.write('Downloaded %d recvideo(s)\r' % count)
                sys.stdout.flush()
                if limit and count >= limit:
                    break
        print('\n[{:.2f} seconds] Done!'.format(time.time() - start_time))

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])