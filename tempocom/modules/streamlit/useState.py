import streamlit as st
from typing import TypeVar, Callable, Tuple

T = TypeVar('T')

def _factory_no_cache(factory_func: Callable[[], T]) -> T:
    return factory_func()

def useState(key: str, value: T = None, factory: Callable[[], T] = None) -> Tuple[T, Callable[[T], None]]:
    """
    React-like useState for Streamlit.
    Returns (value, setValue) for a given key in session_state.
    """
    # Initialize if not already set
    if key not in st.session_state:
        if factory is not None:
            st.session_state[key] = _factory_no_cache(factory)
        else:
            st.session_state[key] = value

    def setState(new_value: T):
        st.session_state[key] = new_value

    return st.session_state[key], setState