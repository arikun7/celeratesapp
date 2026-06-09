import fs from "fs";
import path from "path";

function parseCSV(text: string): string[][] {
  const result: string[][] = [];
  let row: string[] = [];
  let currentVal = '';
  let inQuotes = false;
  
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];
    
    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        currentVal += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      row.push(currentVal.trim());
      currentVal = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      row.push(currentVal.trim());
      result.push(row);
      row = [];
      currentVal = '';
      if (char === '\r' && nextChar === '\n') {
        i++;
      }
    } else {
      currentVal += char;
    }
  }
  if (row.length > 0 || currentVal !== '') {
    row.push(currentVal.trim());
    result.push(row);
  }
  return result.filter(r => r.length > 1 || (r.length === 1 && r[0] !== ''));
}

async function runAudit() {
  const cachePath = path.join(process.cwd(), "dataset_cache.csv");
  let csvText = "";
  if (fs.existsSync(cachePath)) {
    csvText = fs.readFileSync(cachePath, "utf-8");
  } else {
    const spreadsheetUrl = 'https://docs.google.com/spreadsheets/d/1aX6GynZaDd3leFEWfU16wnpxP-PjgyCiaetOkHvCooQ/export?format=csv';
    const response = await fetch(spreadsheetUrl);
    csvText = await response.text();
    fs.writeFileSync(cachePath, csvText, "utf-8");
  }

  const rows = parseCSV(csvText);
  const headers = rows[0].map(h => h.trim());
  
  console.log("=== CSV Audit ===");
  console.log(`Total CSV rows (including header): ${rows.length}`);
  console.log(`Data rows count: ${rows.length - 1}`);
  console.log("Headers:", headers);

  const customerIdIndex = headers.indexOf("Customer_ID");
  const customerNameIndex = headers.indexOf("Customer_Name");
  const ageIndex = headers.indexOf("Age");
  const genderIndex = headers.indexOf("Gender");
  const locationIndex = headers.indexOf("Location");
  const priceIndex = headers.indexOf("Purchase_Amount");
  const freqIndex = headers.indexOf("Frequency_of_Purchase");

  const customers: Record<string, string[]> = {};
  const customerProfiles: Record<string, { name: string, age: string, gender: string, location: string, rowCount: number }> = {};
  
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const cId = r[customerIdIndex];
    if (!cId) continue;
    if (!customers[cId]) {
      customers[cId] = [];
      customerProfiles[cId] = {
        name: r[customerNameIndex] || "",
        age: r[ageIndex] || "",
        gender: r[genderIndex] || "",
        location: r[locationIndex] || "",
        rowCount: 0
      };
    }
    customers[cId].push(`Row ${i}`);
    customerProfiles[cId].rowCount++;
  }

  const uniqueCustomIds = Object.keys(customers);
  console.log(`Unique Customer_IDs: ${uniqueCustomIds.length}`);

  const customerStats = uniqueCustomIds.map(cId => {
    let base = customerProfiles[cId];
    let totalSpend = 0;
    let minRecency = Infinity;
    let count = 0;
    
    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      if (r[customerIdIndex] === cId) {
        totalSpend += parseFloat(r[priceIndex]) || 0;
        
        const dateStr = r[headers.indexOf("Time_of_Purchase")] || '2024-06-01';
        const REFERENCE_DATE = new Date('2024-12-31').getTime();
        const purchaseTime = new Date(dateStr).getTime();
        const diffMs = REFERENCE_DATE - purchaseTime;
        let recencyDays = Math.max(0, Math.floor(diffMs / (1000 * 60 * 60 * 24)));
        if (isNaN(recencyDays)) recencyDays = 120;
        
        if (recencyDays < minRecency) {
          minRecency = recencyDays;
        }
        count++;
      }
    }
    
    return {
      cId,
      name: base.name,
      totalSpend,
      minRecency,
      count
    };
  });

  // Calculate statistics
  const spends = customerStats.map(s => s.totalSpend).sort((a,b) => a - b);
  const frequencies = customerStats.map(s => s.count).sort((a,b) => a - b);
  const recencies = customerStats.map(s => s.minRecency).sort((a,b) => a - b);

  console.log("\n--- AGGREGATED CUSTOMER METRICS ---");
  console.log(`Min/Max/Avg Spend: $${spends[0].toFixed(2)} / $${spends[spends.length-1].toFixed(2)} / $${(spends.reduce((a,b)=>a+b,0)/spends.length).toFixed(2)}`);
  console.log(`Min/Max/Avg Freq (Rows): ${frequencies[0]} / ${frequencies[frequencies.length-1]} / ${(frequencies.reduce((a,b)=>a+b,0)/frequencies.length).toFixed(1)}`);
  console.log(`Min/Max/Avg Recency (Days): ${recencies[0]} / ${recencies[recencies.length-1]} / ${(recencies.reduce((a,b)=>a+b,0)/recencies.length).toFixed(1)}`);
  
  // Percentiles for Spent
  const p20Spend = spends[Math.floor(spends.length * 0.2)];
  const p40Spend = spends[Math.floor(spends.length * 0.4)];
  const p60Spend = spends[Math.floor(spends.length * 0.6)];
  const p80Spend = spends[Math.floor(spends.length * 0.8)];
  console.log(`\nSpend Percentiles (20%, 40%, 60%, 80%): $${p20Spend.toFixed(2)}, $${p40Spend.toFixed(2)}, $${p60Spend.toFixed(2)}, $${p80Spend.toFixed(2)}`);

  // Percentiles for Freq
  const p20Freq = frequencies[Math.floor(frequencies.length * 0.2)];
  const p40Freq = frequencies[Math.floor(frequencies.length * 0.4)];
  const p60Freq = frequencies[Math.floor(frequencies.length * 0.6)];
  const p80Freq = frequencies[Math.floor(frequencies.length * 0.8)];
  console.log(`Frequency Percentiles (20%, 40%, 60%, 80%): ${p20Freq}, ${p40Freq}, ${p60Freq}, ${p80Freq}`);

  // Percentiles for Recency
  const p20Rec = recencies[Math.floor(recencies.length * 0.2)];
  const p40Rec = recencies[Math.floor(recencies.length * 0.4)];
  const p60Rec = recencies[Math.floor(recencies.length * 0.6)];
  const p80Rec = recencies[Math.floor(recencies.length * 0.8)];
  console.log(`Recency Percentiles (20%, 40%, 60%, 80%): ${p20Rec}, ${p40Rec}, ${p60Rec}, ${p80Rec} days`);




  // Check if properties differ for the same customer
  let differingProfilesCount = 0;
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const cId = r[customerIdIndex];
    if (!cId) continue;
    const base = customerProfiles[cId];
    if (r[customerNameIndex] !== base.name || r[ageIndex] !== base.age || r[genderIndex] !== base.gender || r[locationIndex] !== base.location) {
      differingProfilesCount++;
    }
  }
  console.log(`Transactions with differing profile variables for same CustomerID: ${differingProfilesCount}`);

  // Check unique counts of customer identifiers
  console.log("Sample Customers:");
  uniqueCustomIds.slice(0, 5).forEach(cId => {
    console.log(`  ID: ${cId}, Name: ${customerProfiles[cId].name}, Rows containing: ${customerProfiles[cId].rowCount}`);
  });
}

runAudit();
