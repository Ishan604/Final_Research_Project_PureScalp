// upload.js
document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const fileInfo = document.getElementById('fileInfo');
    const imageGrid = document.getElementById('imageGrid');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analysisResults = document.getElementById('analysisResults');
    
    let uploadedFiles = [];
    
    // Browse files button click
    browseBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', function(e) {
        handleFiles(e.target.files);
    });
    
    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropzone.classList.add('active');
    }
    
    function unhighlight() {
        dropzone.classList.remove('active');
    }
    
    dropzone.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });
    
    // Handle uploaded files
    function handleFiles(files) {
        // Only keep one file (the most recent selection/drop first file)
        const list = Array.from(files);
        if (list.length > 0) {
            uploadedFiles = [list[0]];
        }
        updateFileInfo();
        displayImages();
        analyzeBtn.disabled = uploadedFiles.length === 0;
    }
    
    // Update file information display
    function updateFileInfo() {
        if (uploadedFiles.length === 0) {
            fileInfo.textContent = '';
            return;
        }
        
        const totalSize = uploadedFiles[0].size;
        const sizeInMB = (totalSize / (1024 * 1024)).toFixed(2);
        fileInfo.innerHTML = `<p>1 file selected • ${sizeInMB} MB</p>`;
    }
    
    // Display uploaded images
    function displayImages() {
        imageGrid.innerHTML = '';
        
        imageGrid.innerHTML = '';
        const file = uploadedFiles[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {
            const imagePreview = document.createElement('div');
            imagePreview.className = 'image-preview';
            imagePreview.innerHTML = `
                <img src="${e.target.result}" alt="Uploaded image">
                <button class="remove-btn" data-index="0">&times;</button>
            `;
            imageGrid.appendChild(imagePreview);
            const removeBtn = imagePreview.querySelector('.remove-btn');
            removeBtn.addEventListener('click', function() {
                removeImage(0);
            });
        };
        reader.readAsDataURL(file);
    }
    
    // Remove image from upload list
    function removeImage(index) {
        uploadedFiles.splice(index, 1);
        updateFileInfo();
        displayImages();
        analyzeBtn.disabled = uploadedFiles.length === 0;
    }
    
    // Analyze button click
    analyzeBtn.addEventListener('click', async function() {
        if (uploadedFiles.length === 0) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        try {
            const formData = new FormData();
            // send exactly one file
            if (uploadedFiles[0]) {
                formData.append('files', uploadedFiles[0]);
            }

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to analyze images');
            }

            const data = await response.json();
            renderAnalysisResults(data);
        } catch (error) {
            analysisResults.style.display = 'block';
            analysisResults.innerHTML = `
                <div class="result-item">
                    <h4>Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze My Scalp';
            analyzeBtn.disabled = false;
        }
    });

    function renderAnalysisResults(data) {
        analysisResults.style.display = 'block';
        if (!data.success) {
            analysisResults.innerHTML = `
                <div class="result-item">
                    <h4>Error</h4>
                    <p>${data.error || 'Unknown error'}</p>
                </div>
            `;
            return;
        }

        const items = (data.results || []).map((r) => {
            if (r.error) {
                return `
                    <div class="result-item">
                        <h4>${r.filename}</h4>
                        <p>Error: ${r.error}${typeof r.scalp_confidence !== 'undefined' ? ` (scalp confidence: ${(r.scalp_confidence*100).toFixed(1)}%)` : ''}</p>
                    </div>
                `;
            }
            const pct = (r.confidence * 100).toFixed(1);
            return `
                <div class="result-item">
                    <h4>${r.filename}</h4>
                    <p><strong>Predicted:</strong> ${r.predicted_class}</p>
                    <p><strong>Confidence:</strong> ${pct}%</p>
                </div>
            `;
        }).join('');

        analysisResults.innerHTML = `
            <h3>Analysis Results</h3>
            ${items}
            <button class="browse-btn" id="newAnalysisBtn">Start New Analysis</button>
        `;

        document.getElementById('newAnalysisBtn').addEventListener('click', function() {
            resetAnalysis();
        });
    }
    
    // Reset analysis
    function resetAnalysis() {
        uploadedFiles = [];
        updateFileInfo();
        imageGrid.innerHTML = '';
        analysisResults.style.display = 'none';
        analysisResults.innerHTML = '';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze My Scalp';
    }
});