"""Screenshot capture strategies — throughput benchmarking & experimentation.

IMPORTANT: These strategies are NOT used in the production rendering pipeline.
Production rendering goes through:
    pixelrag_render.backends.cdp → pixelrag_render.backends.fast_cdp

The strategies module is a standalone harness for comparing different CDP
capture approaches (scroll timing, parallelism, tiling methods). It was used
during development to converge on the production implementation in backends/.

Production-path strategies (exported, used by bench harness):
- CDPSequentialStrategy: baseline single-tab sequential capture
- CDPDirectClipStrategy: uses CDP Page.captureScreenshot with clip param
- CDPPerTileImgWaitStrategy: waits for image decode between tiles
- CDPOneShotStrategy: full-page capture + local tiling

Experimental strategies (NOT exported, kept for reference):
- cdp_multitab: parallel tabs for concurrent capture
- cdp_overlap: overlapping tile edges for stitching
- cdp_parallel: multi-process parallel rendering
- cdp_dynamic: dynamic tile height based on content
- cdp_pipelined_dc: pipelined directclip with async I/O
- cdp_phased: two-phase (navigate all, then capture all)
- cdp_noscroll: viewport-sized pages (no scrolling)
- cdp_pipelined_tabs: tab-pool pipeline
- cdp_dc_single: single-connection directclip
- cdp_fullpage: Chrome full-page screenshot mode

See docs/screenshot-throughput-optimization.md for benchmark results.
See bench/bench_throughput.py for the harness that runs these.
"""

from .base import CaptureStrategy, TileCapture, ArticleCapture, article_url
from .connection import (
    WebsocketConnection,
    PlaywrightConnection,
    launch_websocket,
    launch_playwright,
)
from .cdp_sequential import CDPSequentialStrategy
from .cdp_directclip import CDPDirectClipStrategy
from .cdp_pertile_imgwait import CDPPerTileImgWaitStrategy
from .cdp_oneshot import CDPOneShotStrategy

__all__ = [
    "CaptureStrategy",
    "TileCapture",
    "ArticleCapture",
    "article_url",
    "WebsocketConnection",
    "PlaywrightConnection",
    "launch_websocket",
    "launch_playwright",
    "CDPSequentialStrategy",
    "CDPDirectClipStrategy",
    "CDPPerTileImgWaitStrategy",
    "CDPOneShotStrategy",
]
