import time
import functools
from typing import Callable, Type, Tuple


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Retry decorator.

    Args:
        max_attempts (int): Maximum number of attempts.
        delay (float): Delay between retries in seconds.
        exceptions (tuple): Exception types to catch and retry on.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_attempts:
                        print(f"❌ Failed after {max_attempts} attempts.")
                        raise

                    print(
                        f"⚠️ Attempt {attempt} failed with error: {e}. "
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(delay)
                    attempt += 1

        return wrapper

    return decorator