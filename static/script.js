document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    if (!form) return; // 非 index 页面无需执行

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const promptContainer = dropZone.querySelector('.drop-zone-prompt');
    
    const predictButton = document.getElementById('predict-button');
    const buttonText = predictButton.querySelector('.mdl-button__text');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    const originalPromptHTML = promptContainer.innerHTML;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        } else {
            dropZone.classList.remove('file-selected');
            promptContainer.innerHTML = originalPromptHTML;
        }
    });

    function handleFileSelect(file) {
        dropZone.classList.add('file-selected');
        promptContainer.innerHTML = `
            <i class="material-icons success-icon">check_circle</i>
            <p><strong>${file.name}</strong></p>
            <p style="font-size: 0.8em; color: #555;">File selected</p>
        `;
    }

    form.addEventListener('submit', function(e) {
        if (fileInput.files.length === 0) {
            alert('Please select a file first!');
            e.preventDefault();
            return;
        }

        if (form.checkValidity()) {
            predictButton.disabled = true;
            buttonText.textContent = 'Predicting...';
            loadingSpinner.classList.add('is-active');
        }
    });
});