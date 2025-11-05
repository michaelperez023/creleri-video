function showVideoModal(){
    document.getElementById('videoLoader').style.display = 'flex';
    document.getElementById('videoLoader').style.flexDirection = 'row';
}

function hideVideoModal(){
    document.getElementById('videoLoader').style.display = 'none';
    
}

function disableFileUploadButton() {
    document.getElementById('file-label').classList.add('disabled'); 
    document.getElementById('file-label').style.opacity = '0.5';
}

function enableFileUploadButton() {
    document.getElementById('file-label').classList.remove('disabled'); 
    document.getElementById('file-label').style.opacity = '1';
}

function conceptVisualizationCollapsable(action){
    conceptVisualizationBox = document.getElementById('concept-visualization-box');
    const treeViewBtn = document.getElementById('tree-view-btn');
    const essentialAttributesBtn = document.getElementById('essential-attributes-btn');
    const leafNodesBtn = document.getElementById('leaf-nodes-btn');
    const collapsiblebtn = document.getElementById('concept-visualization-collapsible-btn');

    const buttons = [treeViewBtn, essentialAttributesBtn, leafNodesBtn];

    if(action){
        if(action === "Collapse"){
            conceptVisualizationBox.style.display = 'none';
            conceptVisualizationBox.classList.remove('expanded');
            collapsiblebtn.innerHTML = '<span class="material-symbols-outlined"> expand_circle_down </span>'
        }
        else if(action === "Expand"){
            conceptVisualizationBox.style.display = '';
            conceptVisualizationBox.classList.add('expanded');
            collapsiblebtn.innerHTML = '<span class="material-symbols-outlined"> expand_circle_up </span>'
        }
    }
    else{
        if(conceptVisualizationBox.style.display === ''){
            conceptVisualizationBox.style.display = 'none';
            conceptVisualizationBox.classList.remove('expanded');
            collapsiblebtn.innerHTML = '<span class="material-symbols-outlined"> expand_circle_down </span>'
        }
        else{
            conceptVisualizationBox.style.display = '';
            conceptVisualizationBox.classList.add('expanded');
            collapsiblebtn.innerHTML = '<span class="material-symbols-outlined"> expand_circle_up </span>'
        }
    }
}

function displayMessage(message, className) {
    var conversationArea = document.getElementById('conversation-area');
    var messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.classList.add('chat-message', className);
    conversationArea.appendChild(messageDiv);
    conversationArea.scrollTop = conversationArea.scrollHeight;
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('file-upload').addEventListener('change', handleFileUpload);
    
    document.getElementById('user-input').addEventListener('keyup', function() {
        this.rows = 1;
        this.rows = Math.min(4,this.scrollHeight/32);
        if (event.key !== 'Enter') {
            this.rows = 1;
            this.rows = Math.min(4,this.scrollHeight/32);
        }
    }, false);

    document.getElementById('segmentation-type-select').addEventListener('change', function () {
        let selectedSegmentationType = this.value;
        console.log("Selected segmentation type:", selectedSegmentationType);

        // Send the selected port to the backend
        fetch('/set_backend_segmentation_type', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ segmentationType: selectedSegmentationType }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Segmentation type successfully updated:', data.type);
            } else {
                console.error('Failed to update segmentation type');
            }
        })
        .catch(error => console.error('Error:', error));
    });
    
});
