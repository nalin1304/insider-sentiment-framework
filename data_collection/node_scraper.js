const { chromium } = require('playwright-extra');
const stealth = require('puppeteer-extra-plugin-stealth')();
const fs = require('fs');

chromium.use(stealth);

async function main() {
    const args = process.argv.slice(2);
    let fromDate = "";
    let toDate = "";
    let outputPath = "";

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--from-date') fromDate = args[i + 1];
        if (args[i] === '--to-date') toDate = args[i + 1];
        if (args[i] === '--output-path') outputPath = args[i + 1];
    }

    if (!fromDate || !toDate || !outputPath) {
        console.error("Missing arguments. Require --from-date, --to-date, --output-path");
        process.exit(1);
    }

    console.log(`Starting stealth download for ${fromDate} to ${toDate}`);
    const browser = await chromium.launch({ headless: true });
    
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport: { width: 1280, height: 720 }
    });
    const page = await context.newPage();

    try {
        console.log("Navigating to NSE homepage to establish session...");
        await page.goto('https://www.nseindia.com', { waitUntil: 'networkidle', timeout: 45000 });
        await page.waitForTimeout(5000); // Wait a bit for cookies/scripts to settle

        const apiUrl = `https://www.nseindia.com/api/corporates-pit?index=equities&from_date=${fromDate}&to_date=${toDate}&csv=true`;
        console.log(`Fetching API data via page navigation to download...`);

        // Setting up a download listener
        const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
        
        await page.evaluate((url) => {
            const link = document.createElement('a');
            link.href = url;
            link.download = '';
            document.body.appendChild(link);
            link.click();
        }, apiUrl);

        const download = await downloadPromise;
        const tempPath = await download.path();
        
        fs.copyFileSync(tempPath, outputPath);
        console.log(`Successfully saved to ${outputPath}`);

    } catch (err) {
        console.error("Error during scraping:", err);
        await browser.close();
        process.exit(1);
    }

    await browser.close();
}

main();
