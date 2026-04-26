"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers, plus
some reference implementations:

* :class:`~.Controller`: The abstract base class defining the interface for all controllers.
"""  # noqa: E501, required for linking in the docs

from ece484_fly.control.controller import Controller

__all__ = ["Controller"]
