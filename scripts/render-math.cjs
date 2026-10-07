// Compile LaTeX to HTML + MathML at build time: no client-side math dependency.
const fs = require('node:fs');
const path = require('node:path');
const katex = require('../assets/vendor/katex/katex.min.js');
const root = path.join(__dirname, '../dist');
let count = 0;
function visit(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes:true})) {
    const p = path.join(dir,entry.name);
    if (entry.isDirectory()) visit(p);
    else if (entry.name.endsWith('.html')) {
      let text = fs.readFileSync(p,'utf8');
      text = text.replace(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g, (_,block,inline) => {
        count++;
        const displayMode = block !== undefined;
        const html = katex.renderToString(displayMode ? block : inline, {displayMode,throwOnError:true,strict:'error',output:'htmlAndMathml',trust:false});
        return displayMode ? `<div class="math-display">${html}</div>` : html;
      });
      fs.writeFileSync(p,text);
    }
  }
}
visit(root);
console.log(`Compiled ${count} equations to accessible HTML + MathML.`);
