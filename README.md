# CLEAR + HOPHY Project Page

Static project webpage for showcasing CLEAR and HOPHY, two large-scale off-road planning systems:

- **CLEAR**: semantic-geometric terrain abstraction for scalable off-road planning.
- **HOPHY**: hierarchical hypergraph planning for repeated queries, replanning, and multi-agent missions.

The page follows an academic project-page style similar to the E3D webpage and is designed for hosting through GitHub Pages or any static web server.

## Contents

```text
CLEAR_HOPHY/
├── index.html
├── README.md
└── static/
    ├── css/
    ├── images/
    ├── js/
    └── videos/
```

Main assets:

- `static/images/clear_hophy/`: CLEAR and HOPHY figures used in the webpage.
- `static/videos/CLEAR_RAL_suppl_video.mp4`: CLEAR supplementary video.
- `static/videos/HOPHY_Overview.mp4`: HOPHY overview video.
- `static/images/ub_logo.png`: University at Buffalo logo.
- `static/images/drones_logo.png`: Drones Lab logo.

## Local Preview

From this directory, run:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

If port `8000` is already in use, choose another port:

```bash
python3 -m http.server 8001
```

## Paper Links

Do not add local paper PDFs to this repository. Add paper links only when a public URL is available.

Current paper links:

- CLEAR: https://arxiv.org/abs/2601.13361
- HOPHY: not linked yet because no public paper URL is currently included.

## Updating Videos

Replace the files in `static/videos/` with the final videos when available. Keep the same filenames if you do not want to update `index.html`.

Expected filenames:

- `CLEAR_RAL_suppl_video.mp4`
- `HOPHY_Overview.mp4`

## Updating Content

Edit `index.html` for page text, layout, links, and figure placement.

Edit `static/css/index.css` for styling changes. The template CSS and JS files under `static/css/` and `static/js/` come from the academic project-page template.

## Deployment

This is a static website. To deploy with GitHub Pages:

1. Commit the `CLEAR_HOPHY/` directory.
2. Configure GitHub Pages to serve the folder or copy its contents to the branch/folder used by the Pages site.
3. Confirm all asset paths remain relative, such as `static/images/...` and `static/videos/...`.

## Attribution

The page structure is based on the Academic Project Page Template:

https://github.com/eliahuhorwitz/Academic-project-page-template
