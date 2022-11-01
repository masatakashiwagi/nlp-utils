"""
Utilities.
"""


def load_data(filepath, encoding='utf-8'):
    """Load ja.text8 from:
    https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/ja.text8.zip

    Args:
      filepath: filename to extract.

    Returns:
      text: a list of words.
    """
    with open(filepath, encoding=encoding) as f:
        return f.read()
