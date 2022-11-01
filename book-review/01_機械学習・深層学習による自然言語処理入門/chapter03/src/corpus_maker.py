import json
import os
from typing import Dict

import requests


class YelpClient:
    """Yelp API Client.

    This class provides some requests to get data.
    Reference : https://www.yelp.com/developers

    """

    def __init__(self, api_key, base_path):
        """Constructor."""
        self.api_key = api_key
        self.base_path = base_path
        self.search_path = self.base_path + '/v3/businesses/search'
        self.businesses_path = self.base_path + '/v3/businesses/'
        self.headers = {'content-type': 'application/json', 'authorization': 'Bearer %s' % self.api_key}

    def get_search_request(self, params: Dict) -> Dict:
        """Get request to the API, search basic information about the business.

        Args:
            params (Dict): The search keywords passed to the API.

        Returns:
            The JSON response from the request.

        """
        res = requests.get(self.search_path, headers=self.headers, params=params)

        # when http status error, raise error message.
        res.raise_for_status()
        results = res.json()

        return results

    def get_businesses_request(self, businesses_id: str, locale: str = 'en_US') -> Dict:
        """Get request to the API, return detailed business content.

        Args:
            businesses_id (str): The ID of the business to query.
            locale (str): The must parameter of locale to query. Defaults to en_US.
                Japanese: ja_JP

        Returns:
            The JSON response from the request.

        """
        businesses_path = self.businesses_path + businesses_id
        params = {'locale': locale}
        res = requests.get(businesses_path, headers=self.headers, params=params)

        # when http status error, raise error message.
        res.raise_for_status()
        results = res.json()

        return results

    def get_businesses_review_request(self, businesses_id: str, locale: str = 'en_US') -> Dict:
        """Get request to the API, return detailed business review content.

        Args:
            businesses_id (str): The ID of the business to query.
            locale (str): The must parameter of locale to query. Defaults to en_US.
                Japanese: ja_JP

        Returns:
            The JSON response from the request.

        """
        businesses_path = self.businesses_path + businesses_id + '/reviews'
        params = {'locale': locale}
        res = requests.get(businesses_path, headers=self.headers, params=params)

        # when http status error, raise error message.
        res.raise_for_status()
        results = res.json()

        return results


def extract_businesses_review(review: Dict) -> Dict:
    """Extract detailed business review content.

    Args:
        review (Dict): The JSON response from get_businesses_review_request method.

    Returns:
        The content of rating and review.

    """
    review_response = review['reviews']
    for review in review_response:
        rating = review.get('rating')
        text = review.get('text')
        data = {
            'rating': rating,
            'comment': text
        }
        yield data


def save_as_json(save_file_path: str, record: Dict):
    """Save data as json."""
    with open(save_file_path, mode='a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """Main function."""
    # set your API KEY.
    API_KEY = None
    # API_KEY = os.environ["YELP_API_KEY"]
    BASE_PATH = 'https://api.yelp.com'

    # 保存先のファイルパスを指定
    save_file_path = ""

    # Yelp Clientをインスタンス化.
    client = YelpClient(API_KEY, BASE_PATH)

    # カテゴリは"ラーメン"にしました
    # "term"と"location"は必須で必要なkey
    params = {"term": "food", "categories": "Ramen", "location": "Tokyo"}
    search_res = client.get_search_request(params)

    # とりあえず、サンプルとして3番目の"カラシビ味噌らー麺-鬼金棒"をチョイス
    businesses_id = search_res['businesses'][2]['id']
    review_res = client.get_businesses_review_request(businesses_id, locale='ja_JP')
    records = extract_businesses_review(review_res)
    for record in records:
        save_as_json(save_file_path, record)


if __name__ == '__main__':
    main()
