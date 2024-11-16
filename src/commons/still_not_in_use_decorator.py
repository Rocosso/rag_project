import warnings
import functools

def still_not_in_use(func):
    """
    Decorador para marcar métodos que aún no están en uso.
    Emite una advertencia cada vez que se llama al método.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"El método '{func.__name__}' no está en uso y puede estar desactualizado. remueva este decorador "
            f"para evitar futuras advertencias.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper