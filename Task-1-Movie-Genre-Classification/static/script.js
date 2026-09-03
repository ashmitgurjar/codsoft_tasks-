document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const plotInput = document.getElementById("plotInput");
    const modelSelect = document.getElementById("modelSelect");
    const predictBtn = document.getElementById("predictBtn");
    const clearBtn = document.getElementById("clearBtn");
    const wordCount = document.getElementById("wordCount");
    const modelBadge = document.getElementById("modelBadge");

    const placeholderState = document.getElementById("placeholderState");
    const loadingState = document.getElementById("loadingState");
    const outputContent = document.getElementById("outputContent");

    const topGenreName = document.getElementById("topGenreName");
    const topGenreConfidence = document.getElementById("topGenreConfidence");
    const probBarsContainer = document.getElementById("probBarsContainer");
    const tagCloud = document.getElementById("tagCloud");

    const samplesGrid = document.getElementById("samplesGrid");
    const benchmarkTableBody = document.getElementById("benchmarkTableBody");
    const genreKeywordsGrid = document.getElementById("genreKeywordsGrid");

    // Color map for genres
    const genreColors = {
        "Action": "#ef4444",
        "Sci-Fi": "#06b6d4",
        "Horror": "#9333ea",
        "Comedy": "#f59e0b",
        "Drama": "#3b82f6",
        "Thriller": "#e11d48",
        "Romance": "#ec4899",
        "Mystery": "#10b981",
        "Animation": "#84cc16"
    };

    // Update word count
    plotInput.addEventListener("input", () => {
        const text = plotInput.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        wordCount.textContent = `${words} word${words !== 1 ? 's' : ''}`;
    });

    // Clear button
    clearBtn.addEventListener("click", () => {
        plotInput.value = "";
        wordCount.textContent = "0 words";
        showPlaceholder();
    });

    // Model selection change
    modelSelect.addEventListener("change", () => {
        modelBadge.textContent = modelSelect.value;
        if (plotInput.value.trim().length > 0) {
            classifyPlot();
        }
    });

    // Predict button click
    predictBtn.addEventListener("click", classifyPlot);

    // Predict function
    async function classifyPlot() {
        const text = plotInput.value.trim();
        if (!text) {
            alert("Please type or select a movie plot summary to classify.");
            return;
        }

        showLoading();

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    model: modelSelect.value
                })
            });

            const data = await response.json();

            if (response.ok && !data.error) {
                renderResults(data);
            } else {
                alert(data.error || "An error occurred during classification.");
                showPlaceholder();
            }
        } catch (err) {
            console.error("Error predicting genre:", err);
            alert("Failed to connect to backend server.");
            showPlaceholder();
        }
    }

    function showPlaceholder() {
        placeholderState.classList.remove("hidden");
        loadingState.classList.add("hidden");
        outputContent.classList.add("hidden");
    }

    function showLoading() {
        placeholderState.classList.add("hidden");
        loadingState.classList.remove("hidden");
        outputContent.classList.add("hidden");
    }

    function renderResults(data) {
        placeholderState.classList.add("hidden");
        loadingState.classList.add("hidden");
        outputContent.classList.remove("hidden");

        // Set top genre
        const primaryGenre = data.top_genre;
        const color = genreColors[primaryGenre] || "#6366f1";

        topGenreName.textContent = primaryGenre;
        topGenreName.style.borderColor = color;
        topGenreName.style.color = color;
        topGenreName.style.backgroundColor = `${color}20`;

        topGenreConfidence.textContent = `${data.top_confidence}%`;

        // Render probability bars
        probBarsContainer.innerHTML = "";
        data.predictions.forEach(pred => {
            const row = document.createElement("div");
            row.className = "prob-row";

            const pColor = genreColors[pred.genre] || "#6366f1";

            row.innerHTML = `
                <div class="prob-meta">
                    <span class="genre-name">${pred.genre}</span>
                    <span class="genre-percentage">${pred.percentage}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: 0%; background-color: ${pColor};"></div>
                </div>
            `;
            probBarsContainer.appendChild(row);

            // Animate bar width
            setTimeout(() => {
                row.querySelector(".prob-fill").style.width = `${pred.percentage}%`;
            }, 50);
        });

        // Render key explainability keywords
        tagCloud.innerHTML = "";
        if (data.key_features && data.key_features.length > 0) {
            data.key_features.forEach(feat => {
                const tag = document.createElement("span");
                tag.className = "keyword-tag";
                tag.innerHTML = `${feat.word} <span class="score">(${feat.tfidf_score})</span>`;
                tagCloud.appendChild(tag);
            });
        } else {
            tagCloud.innerHTML = "<span style='color: var(--text-muted); font-size: 0.85rem;'>No strong vocabulary match found in dictionary.</span>";
        }
    }

    // Load sample plots
    async function loadSamples() {
        try {
            const response = await fetch("/api/samples");
            const samples = await response.json();

            samplesGrid.innerHTML = "";
            samples.forEach(sample => {
                const card = document.createElement("div");
                card.className = "sample-card";
                const pColor = genreColors[sample.expected_genre] || "#6366f1";

                card.innerHTML = `
                    <div class="sample-title">${sample.title}</div>
                    <span class="sample-genre-tag" style="background:${pColor}20; color:${pColor};">${sample.expected_genre}</span>
                    <div class="sample-plot-preview">${sample.plot}</div>
                `;

                card.addEventListener("click", () => {
                    plotInput.value = sample.plot;
                    plotInput.dispatchEvent(new Event("input"));
                    // Scroll smooth to predictor
                    document.getElementById("predictor").scrollIntoView({ behavior: "smooth" });
                    classifyPlot();
                });

                samplesGrid.appendChild(card);
            });
        } catch (err) {
            console.error("Failed to load sample plots:", err);
        }
    }

    // Load benchmarks and top keywords
    async function loadMetrics() {
        try {
            const response = await fetch("/api/metrics");
            const metrics = await response.json();

            if (!metrics.models) return;

            // Render Benchmark Table
            benchmarkTableBody.innerHTML = "";
            for (const [modelName, stats] of Object.entries(metrics.models)) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${modelName}</strong></td>
                    <td><span style="color: var(--accent-emerald); font-weight:700;">${(stats.accuracy * 100).toFixed(2)}%</span></td>
                    <td>${(stats.f1_score * 100).toFixed(2)}%</td>
                    <td>${(stats.precision * 100).toFixed(2)}%</td>
                    <td>${(stats.recall * 100).toFixed(2)}%</td>
                    <td>${(stats.cv_accuracy_mean * 100).toFixed(2)}% ± ${(stats.cv_accuracy_std * 100).toFixed(2)}%</td>
                `;
                benchmarkTableBody.appendChild(tr);
            }

            // Render Top Keywords per Genre
            if (metrics.top_keywords) {
                genreKeywordsGrid.innerHTML = "";
                for (const [genre, keywords] of Object.entries(metrics.top_keywords)) {
                    const card = document.createElement("div");
                    card.className = "genre-kw-card";
                    const gColor = genreColors[genre] || "#6366f1";

                    let pillsHtml = keywords.map(kw => `<span class="kw-pill">${kw}</span>`).join("");

                    card.innerHTML = `
                        <div class="genre-kw-title" style="color:${gColor}">${genre}</div>
                        <div class="kw-pill-list">${pillsHtml}</div>
                    `;

                    genreKeywordsGrid.appendChild(card);
                }
            }
        } catch (err) {
            console.error("Failed to load metrics:", err);
        }
    }

    // Initialize UI data
    loadSamples();
    loadMetrics();
});
