'use strict';

// Smoke test for prescription extraction using any image URL or local file.
// Bypasses Twilio + the Express route — fetches the image and calls GPT-4o directly.
// Useful for tuning the extraction prompt without spinning up the full server.
//
// Usage:
//   node scripts/test-prescription.js <image-url-or-local-path>
//
// Examples:
//   node scripts/test-prescription.js sample-rx.jpg
//   node scripts/test-prescription.js https://example.com/prescription.jpg

require('dotenv').config({ override: true });
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const PRESCRIPTION_PROMPT = `Extract information from this medical prescription image.
Return ONLY a valid JSON object with these exact fields:
{
  "name": "patient full name or null",
  "condition": "primary diagnosis or null",
  "doctor": "prescribing doctor name with Dr. prefix or null",
  "hospital": "hospital/clinic name or null",
  "medications": ["Metformin 500mg — every morning with food", "Linagliptin 5mg — every evening"],
  "labs": ["HbA1c test — complete within 14 days"],
  "followUp": "follow-up instruction or null"
}
Medications MUST be plain strings in the format "Name Dose — timing". Never return objects.
If this is not a readable medical prescription, return {"unreadable": true}.`;

async function main() {
  const input = process.argv[2];
  if (!input) {
    console.error('Usage: node scripts/test-prescription.js <image-url-or-local-path>');
    process.exit(1);
  }

  let base64Image;
  let mimeType = 'image/jpeg';

  if (input.startsWith('http://') || input.startsWith('https://')) {
    console.log('Fetching image from:', input);
    const response = await fetch(input);
    if (!response.ok) throw new Error(`Fetch failed: ${response.status}`);
    mimeType = (response.headers.get('content-type') || 'image/jpeg').split(';')[0];
    base64Image = Buffer.from(await response.arrayBuffer()).toString('base64');
  } else {
    const filePath = path.resolve(input);
    console.log('Reading local file:', filePath);
    base64Image = fs.readFileSync(filePath).toString('base64');
    if (input.toLowerCase().endsWith('.png')) mimeType = 'image/png';
  }

  console.log('Sending to GPT-4o vision...\n---');
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    response_format: { type: 'json_object' },
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: PRESCRIPTION_PROMPT },
        { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64Image}`, detail: 'high' } },
      ],
    }],
    max_tokens: 500,
    temperature: 0,
  });

  const raw = response.choices[0].message.content.trim();
  console.log('Raw GPT-4o response:\n', raw, '\n---');

  const parsed = JSON.parse(raw);
  if (parsed.unreadable) { console.log('GPT-4o: image is unreadable or not a prescription'); process.exit(1); }

  console.log('Extracted successfully:');
  console.log(JSON.stringify(parsed, null, 2));

  console.log('\nConfirmation message that WhatsApp would send:');
  const medList = (parsed.medications || []).map(m => `  • ${m}`).join('\n');
  const doctorLine = parsed.doctor ? ` from ${parsed.doctor}` : '';
  const conditionLine = parsed.condition ? ` for ${parsed.condition}` : '';
  console.log(`Got it! I've noted your prescription${doctorLine}${conditionLine}.\n\nMedications:\n${medList}\n\nSend *hi* to start your care journey! 💚`);
}

main().catch(err => { console.error('Error:', err.message); process.exit(1); });
