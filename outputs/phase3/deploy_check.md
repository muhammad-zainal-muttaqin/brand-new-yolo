# Deploy Check

- Status: **deferred by repo override**.
- TFLite export: `skipped for now`
- TFLite INT8 export: `skipped for now`
- Rationale: amankan final `best.pt` dan validasi metrik eksperimen lebih dulu; konversi deploy boleh dilakukan belakangan sebagai langkah engineering terpisah.
- Important: jika nanti dikonversi ke TFLite / INT8 / format lain, akurasi, ukuran, latency, dan kompatibilitas hardware **wajib divalidasi ulang** pada artefak hasil konversi itu.
- Best weight size MB: `38.63`
- Inference viability nyata di tablet tetap perlu pengujian hardware terpisah bila device tersedia.
