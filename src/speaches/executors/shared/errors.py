class UnsupportedResponseFormatError(ValueError):
    def __init__(self, response_format: str, model: str) -> None:
        super().__init__(f"'{response_format}' response format is not supported for '{model}' model.")
        self.response_format = response_format
        self.model = model
