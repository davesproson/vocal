class CheckException(Exception):
    """
    An exception which may be raised by a CheckError
    """


class NotCheckedError(Exception):
    """
    An exception which may be raised when check properties are
    accessed before checks have been carried out
    """


class ElementDoesNotExist(Exception):
    """
    Raised when an non existant variable is requested by name
    """
