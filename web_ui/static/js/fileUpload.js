// Upload flow: each selected video is sent to /analyze_video, its segmentation task is polled
// until it completes, and the results are drawn as a timeline with per-segment details whose
// activity cards appear once the segment's grounding jobs finish.

const TASK_POLL_MS = 5000;
const GROUNDING_POLL_MS = 2000;
const GROUNDING_TIMEOUT_MS = 10 * 60 * 1000;
// Network errors and gateway errors, including the 503 the server sends while the segmentation
// API is unreachable, are retried this many times in a row before polling gives up.
const TRANSIENT_HTTP_STATUS = [502, 503, 504];
const MAX_TRANSIENT_FAILURES = 12;

let videoCounter = 0;
let viewerUrl = null;
let releaseViewer = null;
let viewerGeneration = 0;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function responseError(response, action) {
    const body = await response.json().catch(() => ({}));
    const message = body.detail || body.error;
    return new Error(`${action} returned HTTP ${response.status}${message ? `: ${message}` : ''}`);
}

async function handleFileUpload(event) {
    const fileInput = event.target;
    const files = Array.from(fileInput.files);
    const videos = files.filter(file => file.type.startsWith('video/'));
    const skipped = files.filter(file => !file.type.startsWith('video/'));

    if (skipped.length > 0) {
        const names = skipped.map(file => file.name).join(', ');
        displayMessage(`Skipped ${names}: only video files are supported.`, 'system-message');
    }
    if (videos.length === 0) {
        fileInput.value = '';
        return;
    }

    // Read once so that changing the dropdown mid-batch cannot mix modes within a batch.
    const segmentationType = document.getElementById('segmentation-type-select').value;
    const counter = document.getElementById('video-processing-counter');
    setUploadEnabled(false);
    showVideoModal();
    displayMessage("Your videos are being processed. Results will appear in the concept visualization box as soon as they're ready.", 'system-message');

    try {
        for (const [index, file] of videos.entries()) {
            const label = `Video ${index + 1} of ${videos.length}`;
            conceptVisualizationCollapsable('Expand');
            counter.textContent = `Processing Video ${index + 1} / ${videos.length}`;
            try {
                const status = await uploadVideoFile(file, segmentationType);
                await renderVideoResults(status.result, file);
                await displayVideoPreview(file);
                displayMessage(`${label} processed.`, 'system-message');
            } catch (error) {
                console.error(`Processing ${file.name} failed:`, error);
                displayMessage(`${label} (${file.name}) failed: ${error.message}`, 'system-message');
            }
        }
    } finally {
        counter.textContent = '';
        hideVideoModal();
        setUploadEnabled(true);
        fileInput.value = '';
    }
}

// Upload one video and wait for its segmentation task; resolves to the completed task status.
async function uploadVideoFile(file, segmentationType) {
    const maxBytes = Number(document.getElementById('file-upload').dataset.maxBytes);
    if (file.size > maxBytes) {
        throw new Error(`the file is larger than the ${Math.floor(maxBytes / 2 ** 20)} MB upload limit`);
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('segmentation_type', segmentationType);
    const response = await fetch('/analyze_video', { method: 'POST', body: formData });
    if (!response.ok) {
        throw await responseError(response, 'Upload');
    }
    const { task_id: taskId } = await response.json();
    if (!taskId) {
        throw new Error('the server did not return a task id');
    }

    let failures = 0;
    while (true) {
        await sleep(TASK_POLL_MS);
        let statusResponse;
        try {
            statusResponse = await fetch(`/task_status/${encodeURIComponent(taskId)}`);
        } catch (error) {
            // The task keeps running while this server restarts or a tunnel to it reconnects.
            if (++failures <= MAX_TRANSIENT_FAILURES) continue;
            throw error;
        }
        if (!statusResponse.ok) {
            if (TRANSIENT_HTTP_STATUS.includes(statusResponse.status) && ++failures <= MAX_TRANSIENT_FAILURES) {
                continue;
            }
            throw await responseError(statusResponse, 'Status check');
        }
        failures = 0;
        const status = await statusResponse.json();
        if (status.status === 'Completed') {
            return status;
        }
        if (status.status === 'Failed') {
            throw new Error(`video processing failed: ${status.error}`);
        }
        if (status.status !== 'Processing') {
            throw new Error(`unexpected task status: ${status.status}`);
        }
    }
}

// Segment times reach the page through several JSON paths, so compare them with a tolerance.
function sameSpan(a, b) {
    const EPS = 1e-2;
    return Array.isArray(a) && Math.abs(a[0] - b[0]) < EPS && Math.abs(a[1] - b[1]) < EPS;
}

// The timeline is scaled to the video's duration. The browser cannot decode every container
// the backend accepts, so fall back to the end of the last segment.
async function videoDuration(file, fallback) {
    const url = URL.createObjectURL(file);
    const probe = document.createElement('video');
    probe.preload = 'metadata';
    try {
        return await new Promise(resolve => {
            probe.onloadedmetadata = () =>
                resolve(Number.isFinite(probe.duration) && probe.duration > 0 ? probe.duration : fallback);
            probe.onerror = () => resolve(fallback);
            probe.src = url;
        });
    } finally {
        URL.revokeObjectURL(url);
    }
}

function clearSegmentHighlight() {
    d3.selectAll("rect")
        .attr("stroke", "none")
        .attr("stroke-width", 0)
        .style("fill-opacity", 1.0);
}

// Make a clickable element reachable with Tab and operable with Enter or Space, like a button.
function activateWithKeyboard(selection) {
    selection
        .attr("tabindex", 0)
        .attr("role", "button")
        .on("keydown", function (event) {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                this.dispatchEvent(new MouseEvent("click", { bubbles: true }));
            }
        });
}

async function renderVideoResults(result, file) {
    const { proposals, results: segmentPredictions, thumbnails } = result ?? {};
    if (!proposals || !segmentPredictions || !thumbnails) {
        throw new Error('the server response is missing results');
    }
    const videoPaths = Object.keys(proposals);
    if (videoPaths.length === 0) {
        throw new Error('no segments were proposed');
    }
    const segments = proposals[videoPaths[0]].filter(
        s => Array.isArray(s?.segment) && s.segment.length === 2
    );

    const videoID = `video-container-${videoCounter++}`;

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
        .attr("aria-label", `Results for ${file.name}`)
        .attr("aria-expanded", "true")
        .html(`<span class="material-symbols-outlined" aria-hidden="true">expand_circle_up</span>`)
        .style("margin-left", "auto")
        .style("border", "none")
        .style("background", "transparent")
        .on("click", function () {
            const content = d3.select(`#${videoID} .video-content`);
            const isCollapsed = content.style("display") === "none";
            content.style("display", isCollapsed ? "block" : "none");
            collapseButton.attr("aria-expanded", String(isCollapsed));
            collapseButton.select("span").text(isCollapsed ? "expand_circle_up" : "expand_circle_down");
        });

    const contentSection = videoContainer.append("div")
        .attr("class", "video-content")
        .style("padding", "10px");

    contentSection.append("a")
        .text("Click Here for Original Video")
        .attr("href", "#")
        .attr("class", "original-video-link")
        .style("display", "block")
        .style("margin-bottom", "10px")
        .on("click", event => {
            event.preventDefault();
            displayVideoPreview(file);
            clearSegmentHighlight();
        });

    const duration = await videoDuration(file, Math.max(1, ...segments.map(s => s.segment[1])));

    // Timeline dimensions
    const timelineWidth = 1000;
    const timelineHeight = 40;
    const timelineMargin = 5;

    const timelineContainer = contentSection.append("svg")
        .attr("width", timelineWidth)
        .attr("height", timelineHeight)
        .style("background", "#000")
        .style("margin-bottom", "20px");

    const segmentColors = segments.map(() =>
        `hsl(${Math.random() * 360}, 70%, 50%)`
    );

    timelineContainer.append("g")
        .selectAll("rect")
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
        .attr("data-segment-index", (_, i) => i)
        .attr("aria-label", d => `Segment ${d.segment[0].toFixed(2)}s - ${d.segment[1].toFixed(2)}s`)
        .call(activateWithKeyboard)
        .on("click", function () {
            const index = this.getAttribute("data-segment-index");
            contentSection.selectAll(".segment-details").style("display", "none");
            contentSection.select(`.segment-details-${index}`).style("display", "block");

            clearSegmentHighlight();
            d3.select(this)
                .attr("stroke", "#fff")
                .attr("stroke-width", 1)
                .style("fill-opacity", 0.5);
        });

    // Lay out every segment's details first so any segment can be opened right away; activity
    // cards are filled in as each segment's grounding jobs finish.
    const pending = segments.map((segment, index) => {
        const [start, end] = segment.segment;
        const thumbnail = thumbnails.find(t => sameSpan(t.segment, segment.segment));

        const detailsContainer = contentSection.append("div")
            .attr("class", `segment-details segment-details-${index}`)
            .style("margin-top", "20px")
            .style("display", "none");

        detailsContainer.append("h4")
            .text(`Segment: ${start.toFixed(2)}s - ${end.toFixed(2)}s`);

        const segmentInfo = detailsContainer.append("div")
            .style("display", "flex")
            .style("flex-direction", "row")
            .style("gap", "15px");

        const thumbnailContainer = segmentInfo.append("div");
        if (thumbnail?.thumbnail_url) {
            thumbnailContainer.append("img")
                .attr("src", thumbnail.thumbnail_url)
                .attr("alt", `Thumbnail for segment ${index}`)
                .style("max-width", "200px")
                .style("display", "block")
                .style("margin-bottom", "10px")
                .style("cursor", "pointer")
                .call(activateWithKeyboard)
                .on("click", () => playSegment(file, start, end));
        } else {
            thumbnailContainer.append("div")
                .text("No thumbnail available")
                .style("margin-bottom", "10px");
        }

        const actionInfoContainer = segmentInfo.append("div")
            .style("display", "flex")
            .style("flex-direction", "row")
            .style("gap", "10px");

        // The worker skips segments it cannot cut, so predictions are matched by time, not index.
        const prediction = segmentPredictions.find(p => sameSpan(p.segment?.segment, segment.segment));
        return { actionInfoContainer, prediction };
    });

    const groundingDeadline = Date.now() + GROUNDING_TIMEOUT_MS;
    for (const { actionInfoContainer, prediction } of pending) {
        await renderActivityPredictions(actionInfoContainer, prediction, groundingDeadline);
    }
}

async function renderActivityPredictions(container, prediction, deadline) {
    if (prediction?.error) {
        container.append("div").text("Analysis of this segment failed.");
        return;
    }
    if (!prediction || !prediction.results) {
        container.append("div").text("No activity predictions for this segment.");
        return;
    }

    // Grounding job ids come back in the order the worker walked the parsed actions.
    const jobIds = prediction.api_response || [];
    const actions = Object.values(prediction.results)
        .flat()
        .map((action, i) => ({ ...action, jobId: jobIds[i] }))
        .sort((a, b) => b.probability - a.probability);

    const ids = actions.map(a => a.jobId).filter(Boolean);
    if (ids.length === 0) {
        container.append("div").text("No activity predictions for this segment.");
        return;
    }

    const statusMap = await waitAllJobs(ids, deadline);
    const statuses = ids.map(id => statusMap[id].status);
    if (!statuses.includes("SUCCESS_VALID")) {
        container.append("div").text(missingPredictionsNote(statuses));
        return;
    }

    let foundValidPrediction = false;
    let actionCounter = 0;
    for (const action of actions) {
        const state = statusMap[action.jobId];
        if (!state || !state.hasVisualization) continue;
        const valid = state.status === "SUCCESS_VALID";
        if (!valid && foundValidPrediction) continue;

        actionCounter++;
        const card = container.append("div")
            .style("margin-bottom", "10px")
            .style("cursor", "pointer")
            .style("border", "solid")
            .style("border-radius", "5px")
            .style("padding", "10px 20px")
            .style("max-width", "250px")
            .call(activateWithKeyboard)
            .on("click", () => displaySegmentActivityPredictionVideo(action.jobId));

        appendField(card, "Activity Prediction", actionCounter);
        appendField(card, "Activity", valid ? action.pattern.action : "I don't know");
        for (const [key, value] of Object.entries(action.pattern)) {
            if (key !== "action") {
                appendField(card, key, value);
            }
        }

        if (valid) {
            const confidence = action.probability >= 0.67 ? 'high'
                            : action.probability > 0.33 ? 'medium'
                            : 'low';
            const label = confidence.charAt(0).toUpperCase() + confidence.slice(1);
            appendField(card, "Confidence", label)
                .select("span")
                .style("color", getConfidenceColor(confidence));
            foundValidPrediction = true;
        }
    }
}

// Why a segment has no activity cards. Only claim that nothing was found when every grounding
// job returned a result, so that a service failure does not read as an empty prediction.
function missingPredictionsNote(statuses) {
    if (statuses.includes("ERROR")) return "Grounding results for this segment could not be loaded.";
    if (statuses.includes("EXCEPTION")) return "Grounding failed for this segment.";
    if (statuses.includes("TIMEOUT")) return "Timed out waiting for grounding results.";
    if (statuses.includes("NOT_FOUND")) return "Grounding results for this segment have expired.";
    return "No activity predictions for this segment.";
}

// One "label: value" line of an activity card. Keys and values are model output, so they are
// inserted as text, never as HTML.
function appendField(card, label, value) {
    const line = card.append("div");
    line.append("strong").text(`${label}:`);
    line.append("span").text(` ${value}`);
    return line;
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

// Play the original video looped over one segment.
function playSegment(file, start, end) {
    displayVideoPreview(file);
    const videoElement = document.getElementById('media-video');
    videoElement.currentTime = start;
    videoElement.addEventListener('timeupdate', () => {
        // The player loops the whole file, so also catch the wrap back to 0 that the last
        // segment hits before its end time is ever exceeded.
        const t = videoElement.currentTime;
        if (t >= end - 0.05 || t < start - 0.25) {
            videoElement.currentTime = start;
        }
    });
}

// Show a File or Blob in the media viewer, replacing whatever is there. Resolves once the
// video can play, once it fails to decode (reported to the user), or once it is replaced.
function displayVideoPreview(source) {
    viewerGeneration++;
    releaseViewer?.();
    if (viewerUrl) {
        URL.revokeObjectURL(viewerUrl);
    }
    viewerUrl = URL.createObjectURL(source);

    const videoElement = document.createElement('video');
    videoElement.id = 'media-video';
    videoElement.src = viewerUrl;
    videoElement.controls = true;
    videoElement.style.width = '100%';
    videoElement.style.height = '380px';
    videoElement.style.objectFit = 'contain';
    videoElement.autoplay = true;
    videoElement.loop = true;

    const settled = new Promise(resolve => {
        releaseViewer = resolve;
        videoElement.addEventListener('loadeddata', () => resolve());
        videoElement.addEventListener('error', () => {
            if (videoElement.isConnected) {
                displayMessage(`This browser cannot play ${source.name ?? 'the selected video'}.`, 'system-message');
            }
            resolve();
        });
    });

    const existingMediaVideo = document.getElementById('media-video');
    if (existingMediaVideo) {
        existingMediaVideo.replaceWith(videoElement);
    } else {
        document.getElementById('media-player').appendChild(videoElement);
    }
    return settled;
}

async function displaySegmentActivityPredictionVideo(jobId) {
    const generation = ++viewerGeneration;
    try {
        const response = await fetch(`/video/get_result/${encodeURIComponent(jobId)}`);
        if (!response.ok) {
            throw await responseError(response, 'The activity video request');
        }
        const blob = await response.blob();
        // A slow download must not replace something the user has selected since.
        if (generation === viewerGeneration) {
            await displayVideoPreview(blob);
        }
    } catch (error) {
        displayMessage(`Could not load the activity video: ${error.message}`, 'system-message');
    }
}

// Poll one grounding job until it leaves PENDING. Never throws, so one expired or failing job
// cannot stop the rest of the video from rendering.
async function waitForGroundingJob(jobId, deadline) {
    let failures = 0;
    while (true) {
        try {
            const response = await fetch(`/video/result/${encodeURIComponent(jobId)}`);
            if (response.ok) {
                failures = 0;
                const job = await response.json();
                if (job.status !== "PENDING") {
                    return { status: job.status, hasVisualization: job.has_visualization };
                }
            } else if (response.status === 404) {
                // The grounding service forgets jobs when it restarts or once their results expire.
                return { status: "NOT_FOUND", hasVisualization: false };
            } else if (!TRANSIENT_HTTP_STATUS.includes(response.status) || ++failures > MAX_TRANSIENT_FAILURES) {
                return { status: "ERROR", hasVisualization: false };
            }
        } catch (error) {
            if (++failures > MAX_TRANSIENT_FAILURES) {
                return { status: "ERROR", hasVisualization: false };
            }
        }
        if (Date.now() > deadline) {
            return { status: "TIMEOUT", hasVisualization: false };
        }
        await sleep(GROUNDING_POLL_MS);
    }
}

// Resolve a set of job ids concurrently into {jobId: {status, hasVisualization}}.
async function waitAllJobs(jobIds, deadline) {
    const unique = Array.from(new Set(jobIds));
    const states = await Promise.all(unique.map(id => waitForGroundingJob(id, deadline)));
    return Object.fromEntries(unique.map((id, i) => [id, states[i]]));
}
