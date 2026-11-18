const { chromium } = require('../scratch/qa-browser/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require('../frontend/node_modules/typescript');
const source = fs.readFileSync('frontend/lib/characterData.ts', 'utf8');
const charactersModule = { exports: {} };
new Function('exports', ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText)(charactersModule.exports);
const characters = charactersModule.exports.CHARACTERS;
const appearances = JSON.parse(fs.readFileSync('frontend/public/static-data/characters-buod.json', 'utf8'));
const sentence = 'Basahin ang Buod at Buong Kwento, tuklasin ang mga tauhan, at hanapin ang kaugnay na bahagi ng orihinal na akda.';
(async () => {
  const browser = await chromium.launch();
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      const errors = [], failures = [], backend = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('response', r => { if (r.status() >= 400) failures.push(r.url()); });
      page.on('request', r => { if (/\/backend|hf\.space|supabase/.test(r.url())) backend.push(r.url()); });
      await page.goto(process.env.QA_BASE_URL);
      await page.locator('[data-chapter-card]').first().waitFor();
      assert.equal(await page.locator('.literary-hero-copy p').innerText(), sentence);
      assert.equal(await page.locator('[data-literary-search]').getAttribute('placeholder'), 'Maghanap sa nobela...');
      for (const novel of ['noli', 'fili']) {
        await page.locator('.literary-book-' + novel).click();
        await page.locator('#tab-characters').click();
        await page.locator('[data-character-card]').first().waitFor();
        const expected = characters.filter(c => (c.novel === novel || c.novel === 'both') && c.isMajor !== false);
        assert.deepEqual(await page.locator('[data-character-card] h3').allTextContents(), expected.map(c => c.name));
        await page.locator('[data-character-card]').first().click();
        const modal = page.locator('[data-item-modal]');
        await modal.waitFor();
        async function checkLabel() {
          const label = modal.locator('[data-novel-label]');
          assert.equal(await label.innerText(), novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo');
          const style = await label.evaluate(el => { const s = getComputedStyle(el); return { bg:s.backgroundColor, border:s.borderTopWidth, radius:s.borderRadius, spacing:s.letterSpacing, children:el.children.length }; });
          assert.equal(style.bg, 'rgba(0, 0, 0, 0)'); assert.equal(style.border, '0px');
          assert.equal(style.radius, '0px'); assert.ok(['normal','0px'].includes(style.spacing)); assert.equal(style.children, 0);
          assert.equal(await page.getByRole('button', { name: /I-?RANK|NAKA-RANK/i }).count(), 0);
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        }
        await checkLabel();
        const expectedChapters = appearances[expected[0].id].filter(c => c.book === novel || c.book === (novel === 'fili' ? 'elfili' : 'noli')).map(c => c.chapter_number);
        await page.waitForFunction(n => [...document.querySelectorAll('[data-item-modal] h4')].some(el => el.textContent.includes(`Mga Kabanata (${n})`)), expectedChapters.length);
        const numbers = (await modal.getByText(/^Kabanata \d+$/).allTextContents()).map(t => Number(t.match(/\d+/)[0]));
        assert.deepEqual(numbers, expectedChapters);
        await modal.getByRole('button', { name:'Mga Paksa', exact:true }).click();
        await modal.getByRole('button', { name:'Mga Pangungusap', exact:true }).click();
        await modal.getByRole('button', { name:'Isara ang modal', exact:true }).filter({visible:true}).click();
        await modal.waitFor({state:'hidden'});
        await page.locator('#tab-themes').click();
        await page.locator('[data-theme-card]').first().click();
        await modal.waitFor(); await checkLabel();
        await modal.getByRole('button', { name:'Isara ang modal', exact:true }).filter({visible:true}).click();
        await modal.waitFor({state:'hidden'});
      }
      await page.locator('[data-literary-search]').fill('kalayaan');
      await page.locator('[data-literary-search]').press('Enter');
      await page.waitForURL(/\/search\?q=kalayaan/);
      await page.locator('[data-search-result]').first().waitFor();
      assert.equal(await page.locator('[data-literary-search]').getAttribute('placeholder'), 'Maghanap sa nobela...');
      assert.deepEqual(errors, []); assert.deepEqual(failures, []); assert.deepEqual(backend, []);
      console.log('PASS focused UI', width, 'both novels, ordering, labels, no ranking button, search');
      await page.close();
    }
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
