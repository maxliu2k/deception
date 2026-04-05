import os

path = 'app/static/index.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Inject Three.js script
target_a = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />'
if target_a in content and 'three.min.js' not in content:
    content = content.replace(
        target_a,
        target_a + '\n    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    )

# Inject container
target_b = '<div id="transcriptLabel"'
if target_b in content and 'avalon3dContainer' not in content:
    content = content.replace(
        target_b,
        '<div id="avalon3dContainer" class="hidden" style="width: 100%; height: 350px; background: #0b0f19; border-radius: 8px; overflow: hidden; margin-bottom: 12px; position: relative;"></div>\n          ' + target_b
    )

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
