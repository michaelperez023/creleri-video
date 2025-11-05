var OriginalVideoFiles= [];
var FileCounter = 0;

async function handleFileUpload(event) {
    const fileInput = event.target;
    const files = Array.from(event.target.files);
    const unique_file_types = [];

    // Identify unique file types asynchronously
    for (const file of files) {
        if (file.type.startsWith('video/') && !unique_file_types.includes('video')) {
            unique_file_types.push('video');
        } else if (file.type.startsWith('image/') && !unique_file_types.includes('image')) {
            unique_file_types.push('image');
        }
    }

    // Check if no files are selected or if invalid file conditions are met
    if (files.length === 0 || 
        unique_file_types.length > 1 || 
        (files[0].type.startsWith('image/') && files.length > 1)) {
        enableFileUploadButton();
        return; // Exit if no files or invalid file conditions
    }

    if (files[0].type.startsWith('video/')) {
        displayMessage("Your videos are being processed. Results will appear in the concept visualization box as soon as they're ready.", 'system-message');
        // Multiple video uploads
        for (let index = 0; index < files.length; index++) {
            const file = files[index];
            
            if (file.type.startsWith('video/')) {
                const formData = new FormData();
                formData.append('file', file);
                
                conceptVisualizationCollapsable("Expand");
                document.getElementById('video-processing-counter').innerHTML = `Processing Video ${index +1} / ${files.length}`;
                await uploadVideoFile(formData, file); // Await each video upload
                await displayVideoPreview(file);

                displayMessage(`Video ${index + 1} of ${files.length} processed.`, 'system-message');
            }
        }
    } else {
        console.error('Unsupported file type selected.');
    }
    document.getElementById('video-processing-counter').innerHTML = ``;
    fileInput.value = '';
}

async function uploadVideoFile(formData, file) {
    disableFileUploadButton();
    showVideoModal();
    
    console.log(`file(s) uploaded at ${new Date().toLocaleString()}`);

    try {
        const response = await fetch('/analyze_video', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Error uploading video: ${response.statusText}`);
        }

        const { task_id } = await response.json();

        let taskStatus = 'Processing';
        while (taskStatus === 'Processing') {
            await new Promise(resolve => setTimeout(resolve, 5000)); // Wait for 5 seconds
            const statusResponse = await fetch(`/task_status/${task_id}`);
            const statusResult = await statusResponse.json();
            taskStatus = statusResult.status;
            if (taskStatus === 'Completed') {
                console.log('Received response from server:', statusResult);
                await handleVideoUploadSuccess(statusResult, file);
                console.log('Success in handleVideoUploadSuccess:', statusResult);
            } else if (taskStatus === 'Failed') {
                throw new Error(`Video processing failed: ${statusResult.error}`);
            }
        }

        return;
    } catch (error) {
        console.error('Error in uploadVideoFile:', error);
        throw error;

    } finally {
        console.log(`uploadVideoFile finished at ${new Date().toLocaleString()}`);
        
        enableFileUploadButton();
        hideVideoModal();
    }
}

async function handleVideoUploadSuccess(response, file) {
    console.log('Video uploaded successfully', response);
    OriginalVideoFiles.push(file);

    // Ensure response has the expected structure
    if (!response || response.status !== 'Completed' || !response.result) {
        console.error('Unexpected response format', response);
        displayMessage("Error processing video: Invalid response format.", 'system-message');
        hideModal();
        return;
    }

    const { proposals, results: segments_pred_data, thumbnails } = response.result;

    if (!proposals || !segments_pred_data || !thumbnails) {
        console.error('Missing data in response', response);
        displayMessage("Error processing video: Missing data in response.", 'system-message');
        hideModal();
        return;
    }

    const videoPaths = Object.keys(proposals);
    if (videoPaths.length === 0) {
        console.error('No proposals found');
        displayMessage("Error processing video: No proposals found.", 'system-message');
        hideModal();
        return;
    }
    const videoPath = videoPaths[0];
    const segments = proposals[videoPath];

    const videoID = `video-container-${FileCounter}`; // Unique ID for this video container
    const currentFileIndex = FileCounter; // Preserve current file index for later
    FileCounter++; // Increment for the next video

    // Create a new div for the video
    const conceptBox = d3.select("#concept-visualization-box");
    const videoContainer = conceptBox.append("div")
        .attr("class", "video-container")
        .attr("id", videoID)
        .style("border", "solid")
        .style("border-radius", "5px")
        .style("margin", "10px 20px");

    // Collapsible header
    const header = videoContainer.append("div")
        .style("display", "flex")
        .style("justify-content", "space-between")
        .style("align-items", "center")
        .style("background", "#0D082C")
        .style("color", "white")
        .style("padding", "10px");

    header.append("strong").text(`Video: ${file.name}`);

    const collapseButton = header.append("button")
        .html(`<span class="material-symbols-outlined">expand_circle_up</span>`)
        .style("margin-left", "auto")
        .style("border", "none")
        .style("background", "transparent")
        .on("click", function () {
            const content = d3.select(`#${videoID} .video-content`);
            const isCollapsed = content.style("display") === "none";
            content.style("display", isCollapsed ? "block" : "none");
            collapseButton.html(isCollapsed ? `<span class="material-symbols-outlined">expand_circle_up</span>` : `<span class="material-symbols-outlined">expand_circle_down</span>`);
        });

    // Content section
    const contentSection = videoContainer.append("div")
        .attr("class", "video-content")
        .style("padding", "10px");

    // Link to play the original video
    const originalVideoLink = contentSection.append("a")
        .text("Click Here for Original Video")
        .attr("href", "#")
        .attr("class", "original-video-link")
        .style("display", "block")
        .style("margin-bottom", "10px")
        .node();

    originalVideoLink.dataset.fileIndex = currentFileIndex; // Store the file index
    originalVideoLink.addEventListener("click", function (event) {
        event.preventDefault();
        displayVideoPreview(OriginalVideoFiles[originalVideoLink.dataset.fileIndex]);
        
        d3.selectAll("rect")
        .attr("stroke", "none")
        .attr("stroke-width", 0)
        .style("fill-opacity", 1.0);
    });

    // Calculate the duration of the video
    const videoElement = document.createElement("video");
    const videoURL = URL.createObjectURL(file);

    const duration = await new Promise((resolve, reject) => {
        videoElement.src = videoURL;
        videoElement.preload = "metadata";

        videoElement.onloadedmetadata = () => {
            resolve(videoElement.duration);
            URL.revokeObjectURL(videoURL); // Clean up the URL
        };

        videoElement.onerror = (e) => {
            reject("Error loading video metadata: " + e);
        };
    });

    console.log(`Calculated Video Duration: ${duration}s`);

    // Timeline dimensions
    const timelineWidth = 1000;
    const timelineHeight = 40;
    const timelineMargin = 5;

    // SVG container for the timeline
    const timelineContainer = contentSection.append("svg")
        .attr("width", timelineWidth)
        .attr("height", timelineHeight)
        .style("background", "#000")
        .style("margin-bottom", "20px");

    // Generate unique colors for segment overlays
    const segmentColors = segments.map(() =>
        `hsl(${Math.random() * 360}, 70%, 50%)`
    );

    // Draw segment markers
    const segmentsGroup = timelineContainer.append("g");

    segmentsGroup.selectAll("rect")
        .data(segments)
        .enter()
        .append("rect")
        .attr("x", d => (d.segment[0] / duration) * timelineWidth)
        .attr("y", timelineMargin)
        .attr("width", d => Math.max(((d.segment[1] - d.segment[0]) / duration) * timelineWidth, 1))
        .attr("height", timelineHeight - 2 * timelineMargin)
        .attr("fill", (_, i) => segmentColors[i])
        .attr("stroke", "none")
        .attr("stroke-width", 0)
        .style("cursor", "pointer")
        .attr("data-segment-index", (_, i) => i);

    console.log('Segments:', segments);
    console.log('Thumbnails:', thumbnails);

    const segmentDataList = segments.map((segment, index) => {
      // tolerate small float drift when matching by times
      const EPS = 1e-2;
      let matchingThumbnail = thumbnails.find(t =>
        t.segment && segment.segment &&
        Math.abs(t.segment[0] - segment.segment[0]) < EPS &&
        Math.abs(t.segment[1] - segment.segment[1]) < EPS
      );

      // fallback by index if no time match
      if (!matchingThumbnail) matchingThumbnail = thumbnails[index] || null;

      const segment_pred_data = segments_pred_data[index];

      return {
        segment,
        thumbnail_url: matchingThumbnail?.thumbnail_url ?? null,
        segment_pred_data,
      };
    });

    // Pre-generate segment details for each segment
    for (const [index, segmentData] of segmentDataList.entries()) {
        const segment = segmentData.segment;
        const thumbnail_url = segmentData.thumbnail_url;
        const segment_pred_data = segmentData.segment_pred_data;

        const detailsContainer = contentSection.append("div")
            .attr("class", `segment-details segment-details-${index}`)
            .style("margin-top", "20px")
            .style("display", "none");

        const start = segment.segment[0];
        const end = segment.segment[1];

        if (segment.segment && segment.segment.length === 2) {
            detailsContainer.append("h4")
                .text(`Segment: ${start.toFixed(2)}s - ${end.toFixed(2)}s`);
        } else {
            detailsContainer.append("h4").text("Invalid Segment Data");
            continue;
        }

        const segmentInfo = detailsContainer.append("div")
            .style("display", "flex")
            .style("flex-direction", "row")
            .style("gap", "15px");

        // Thumbnail container
        const thumbnailContainer = segmentInfo.append("div");

        // Display the thumbnail image
        if (thumbnail_url) {
            thumbnailContainer.append("img")
                .attr("src", thumbnail_url)
                .attr("alt", `Thumbnail for segment ${index}`)
                .style("max-width", "200px")
                .style("display", "block")
                .style("margin-bottom", "10px")
                .style("cursor", "pointer")
                .on("click", () => {
                    // When the thumbnail is clicked, seek the video to the start of the segment
                    displayVideoPreview(OriginalVideoFiles[originalVideoLink.dataset.fileIndex]);
                    seekVideoTo(start);
                    let videoElement = document.getElementById('media-video');

                    videoElement.addEventListener('timeupdate', () => {
                        if (videoElement.currentTime > end) {
                            videoElement.currentTime = start;
                            videoElement.play();
                        }
                    });
                });
        } else {
            thumbnailContainer.append("div")
                .text("No thumbnail available")
                .style("margin-bottom", "10px");
        }

        // Action info container
        const actionInfoContainer = segmentInfo.append("div")
            .style("display", "flex")
            .style("flex-direction", "row")
            .style("gap", "10px");

        if (!segment_pred_data || !segment_pred_data.results) {
            actionInfoContainer.append("div").text("No activity predictions for this segment.");
            continue;
        }

        // Process the predictions and attach job ids to each action
        const api_responses = segment_pred_data.api_response || [];
        let api_index = 0;

        for (const action_key in (segment_pred_data.results || {})) {
            segment_pred_data.results[action_key].forEach((action) => {
            if (api_index < api_responses.length) {
                action.video_api_id = api_responses[api_index];
                api_index++;
            }
            });
        }
        
        const allActions = Object.values(segment_pred_data.results || {}).flat();
        allActions.sort((a, b) => b.probability - a.probability);
        
        const jobIds = allActions.map(a => a.video_api_id).filter(Boolean);
        if (jobIds.length === 0) {
            actionInfoContainer.append("div").text("No activity predictions for this segment.");
            continue;
        }
        
        const statusMap = await waitAllJobs(jobIds, { pollMs: 2000, timeoutMs: 10 * 60 * 1000 });
        
        const any_valid = jobIds.some(id => statusMap[id]?.status === "SUCCESS_VALID");
        if (!any_valid) {
            actionInfoContainer.append("div").text("No activity predictions for this segment.");
            continue;
        }

        let found_valid_prediction = false;
        let action_counter = 0;
        
        for (const actionData of allActions) {
            const state = statusMap[actionData.video_api_id];
            if (!state) continue;

            const confidence = actionData.probability >= 0.67 ? 'high'
                            : actionData.probability > 0.33 ? 'medium'
                            : 'low';
            const confidenceColor = getConfidenceColor(confidence);

            const { status: Action_Status, hasVisualization } = state;

            if (actionData && hasVisualization && (Action_Status === "SUCCESS_VALID" || !found_valid_prediction)) {
                action_counter++;

                const actionInfo = actionInfoContainer.append("div")
                    .style("margin-bottom", "10px")
                    .style("cursor", "pointer")
                    .style("border", "solid")
                    .style("border-radius", "5px")
                    .style("padding", "10px 20px")
                    .style("max-width", "250px")
                    .on("click", () => {
                        displaySegmentActivityPredictionVideo(actionData.video_api_id);
                    });

                actionInfo.html(`
                  <div><strong>Activity Prediction:</strong> ${action_counter}</div>
                  <div><strong>Activity:</strong> ${Action_Status === "SUCCESS_VALID" ? actionData.pattern.action : "I don't know"}</div>
                `);

                for (const [key, value] of Object.entries(actionData.pattern)) {
                    if (key !== "action") {
                    actionInfo.append("div").html(`<strong>${key}:</strong> ${value}`);
                    }
                }

                if (Action_Status === "SUCCESS_VALID") {
                    actionInfo.append("div").html(
                        `<strong>Confidence:</strong> <span style="color:${confidenceColor};">${confidence.charAt(0).toUpperCase() + confidence.slice(1)}</span>`
                    );
                    found_valid_prediction = true;
                }
            }
        }

        // Add click handlers to toggle visibility
        segmentsGroup.selectAll("rect").on("click", function () {
            const index = d3.select(this).attr("data-segment-index");
            contentSection.selectAll(".segment-details").style("display", "none");
            contentSection.select(`.segment-details-${index}`).style("display", "block");

            d3.selectAll("rect")
            .attr("stroke", "none")
            .attr("stroke-width", 0)
            .style("fill-opacity", 1.0);

            // Highlight the clicked rectangle
            d3.select(this)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1)
            .style("fill-opacity", 0.5);
        });
    }
}

// Helper function to get color based on confidence level
function getConfidenceColor(confidence) {
    if (confidence === 'high') {
        return 'green';    // High confidence
    } else if (confidence === 'medium') {
        return 'orange';   // Medium confidence
    } else if (confidence === 'low') {
        return 'red';      // Low confidence
    } else {
        return 'gray';     // Unknown confidence
    }
}

// Function to seek video to the specified time
function seekVideoTo(time) {
    const videoElement = document.getElementById('media-video');
    if (videoElement) {
        videoElement.currentTime = time;
        videoElement.play();  // Automatically play after seeking
    }
}

async function displayVideoPreview(file) {
    return new Promise((resolve, reject) => {
        const mediaContainer = document.getElementById('media-player');
        const videoElement = document.createElement('video');
        videoElement.id = 'media-video';
        videoElement.src = URL.createObjectURL(file);
        videoElement.controls = true;
        videoElement.style.width = '100%';
        videoElement.style.height = '380px';
        videoElement.style.objectFit = 'contain';
        videoElement.autoplay = true;
        videoElement.loop = true;

        // Listen for the 'loadeddata' event to ensure the video is ready
        videoElement.addEventListener('loadeddata', () => {
            resolve();  // Resolve the promise when the video is ready
        });

        // Handle any potential loading errors
        videoElement.addEventListener('error', (error) => {
            reject(error);  // Reject the promise on load error
        });

        // Replace the existing video or append a new one
        const existingMediaVideo = document.getElementById('media-video');
        if (existingMediaVideo) {
            existingMediaVideo.replaceWith(videoElement);
        } else {
            mediaContainer.appendChild(videoElement);
        }
    });
}

async function displaySegmentActivityPredictionVideo(video_uuid) {
    try {
        const response = await fetch(`/video/get_result/${video_uuid}`, {
            method: 'GET'
        });

        // Check if the response is okay (status 200)
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // Convert the response to a blob
        const blob = await response.blob();

        // Call displayVideoPreview with the video URL
        displayVideoPreview(blob);
    } catch (error) {
        console.error('Error fetching segmented video:', error);
        return null;
    }
}

// Poll until the job leaves PENDING (or times out)
async function waitJobUntilDone(jobId, {pollMs = 2000, timeoutMs = 10 * 60 * 1000} = {}) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    const res = await fetch(`/video/result/${jobId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const status = data.status;                 // "PENDING", "SUCCESS_VALID", "SUCCESS_INVALID", "EXCEPTION", "CANCELLED"
    const hasVisualization = data.has_visualization;

    if (status !== "PENDING") return {status, hasVisualization};
    if (Date.now() > deadline) return {status: "TIMEOUT", hasVisualization: false};

    await new Promise(r => setTimeout(r, pollMs));
  }
}

// Batch helper to resolve a set of IDs concurrently
async function waitAllJobs(jobIds, opts) {
  const uniq = Array.from(new Set(jobIds.filter(Boolean)));
  const pairs = await Promise.all(
    uniq.map(async id => [id, await waitJobUntilDone(id, opts)])
  );
  return Object.fromEntries(pairs); // return a POJO so statusMap[id] works
}