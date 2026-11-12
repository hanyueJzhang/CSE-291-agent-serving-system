"""Report table renderers."""

from . import mooncake_summary


RENDERERS = {
    "mooncake_summary": mooncake_summary.render,
}
