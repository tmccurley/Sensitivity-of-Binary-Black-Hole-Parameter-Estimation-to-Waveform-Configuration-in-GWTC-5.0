# Website source

`index.html` and `about.html` are the canonical website pages. `styles.css` supplies the shared black-hole visual system, while `site-shell.js` adds reduced-motion-aware page transitions and scroll feedback. The main visualization includes D3, explanatory text, and a complete embedded-data fallback, so it remains usable as a static GitHub Pages site.

`assets/waveform-comparison-data.js` is refreshed by `python-code/site-tools/launch_dashboard.py`. When present, this generated file overrides the embedded confirmatory dataset before the charts initialize; if it is unavailable, the embedded fallback still allows the page to render.
