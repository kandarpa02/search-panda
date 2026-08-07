class APIError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)


class URLError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)