"""Compatibility entry point for the private, non-adjudicating case renderer."""

from trustretina.data.review_render import render, main

if __name__ == "__main__":
    raise SystemExit(main())
