"""Comparative report figure renderers."""

from . import capacity, concurrency, context_growth, host_cache, tool_delay


RENDERERS = {
    "capacity": capacity.render,
    "concurrency": concurrency.render,
    "context_growth": context_growth.render,
    "tool_delay": tool_delay.render,
    "host_cache": host_cache.render,
}


