---
name: agentkit
description: CLI toolkit for inspecting, rendering and processing images, PDF and DOCX files. Prefer these tools over writing custom scripts for supported file operations.
---

# AgentKit

Use AgentKit for file inspection, visual analysis and routine file processing. Avoid implementing custom scripts for operations already supported by AgentKit.

## Tool discovery

AgentKit exposes a hierarchical CLI. Discover available commands and their arguments using `--help`:

```bash
agentkit --help
agentkit image --help
agentkit pdf --help
agentkit docx --help
agentkit docx structure --help
```

Consult command-specific help when unsure about arguments. Do not invent unsupported flags.

## Tool selection

### Images

- `image info` — inspect image properties.
- `image crop` — inspect a specific image region.
- `image resize` — resize an image while preserving its aspect ratio.
- `image rotate` — rotate an image clockwise by 90, 180 or 270 degrees.
- `image annotate` — add geometric annotations.
- `image grid` — overlay a coordinate grid with optional labels for visual inspection.
- `image find` — locate connected regions of a reference color and write a JSON analysis artifact.
- `image sample` — inspect exact pixels, 3x3 neighborhoods and regional color statistics.
- `image sheet` — combine multiple images for visual inspection.

### PDF

- `pdf info` — inspect document metadata.
- `pdf text` — extract the available text layer.
- `pdf render` — render pages for visual inspection.

### DOCX

- `docx structure` — inspect document content, formatting, tables and embedded images.
- `docx render` — inspect actual page layout. **Windows only; requires Microsoft Word.**

Select the appropriate structure format:

- `text` — content analysis with minimal context consumption.
- `dsl` — structural and formatting analysis by an LLM.
- `json` — programmatic processing.

## Document analysis

Prefer text extraction for content analysis and page rendering for visual analysis.

When analyzing a document template, inspect its structure and formatting before generating a new document.

After generating a document, render and visually inspect it for layout defects, clipping, incorrect pagination and misplaced elements.

```bash
agentkit docx structure report.docx --out structure --format dsl
agentkit docx render report.docx --out pages
agentkit pdf render report.pdf --out page.png --page 1
```

Open rendered images using the available image-viewing tool. Use contact sheets for overview and individual pages for detailed inspection.

## Output handling

- Check the process exit code.
- Read the JSON response from stdout on success.
- Inspect stderr on failure.
- Follow returned paths to access generated files and manifests.
- Use unique output paths. Existing files are not overwritten.
- Avoid repeated extraction or rendering when previous results remain valid.
- Do not load large structural outputs into context unless necessary.

## Limitations

- Maximum file size: 100 MiB.
- Maximum raster image size: 50 million pixels.
- Image transformations operate on stored pixels: EXIF Orientation is not applied automatically.
- Output metadata, including EXIF, ICC and DPI, is not preserved.
- DOCX rendering is currently Windows-only and requires Microsoft Word. Use trusted documents only.
- Document generation, editing and specialized operations remain the agent's responsibility.