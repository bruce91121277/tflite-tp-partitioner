class ToolError(Exception):
    """An actionable, fail-closed tool error."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__("{}: {}".format(code, message))


class TFLiteDependencyError(ToolError):
    pass


class ModelValidationError(ToolError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__(
            "MODEL_VALIDATION_FAILED",
            "; ".join("{}: {}".format(code, message) for code, message in self.issues),
        )
