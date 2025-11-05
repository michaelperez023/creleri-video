// Wait for the DOM to be loaded before initialising the media player
document.addEventListener("DOMContentLoaded", function() { 
    initialiseMediaPlayer(); 
    EnableDisableVideoControls();

}, false);


// Variables to store handles to various required elements
var mediaPlayer;
var playPauseBtn;
var muteBtn;
var progressBar;
var progress;
var flag = 0;


function initialiseMediaPlayer() {
	// Get a handle to the player
	mediaPlayer = document.getElementById('media-video');
	
	// Get handles to each of the buttons and required elements
	playPauseBtn = document.getElementById('play-pause-button');
	muteBtn = document.getElementById('mute-button');
	progressBar = document.getElementById('progress-bar');
	progress=document.getElementById('progress');

	// Hide the browser's default controls
	 mediaPlayer.controls = false;
	
	// Add a listener for the timeupdate event so we can update the progress bar
	mediaPlayer.addEventListener('timeupdate', updateProgressBar, false);
	
	// Add a listener for the play and pause events so the buttons state can be updated
	mediaPlayer.addEventListener('play', function() {
		// Change the button to be a pause button
		changeButtonType(playPauseBtn, 'pause');
	}, false);
	mediaPlayer.addEventListener('pause', function() {
		// Change the button to be a play button
		changeButtonType(playPauseBtn, 'play');
	}, false);
	
	// need to work on this one more...how to know it's muted?
	mediaPlayer.addEventListener('volumechange', function(e) { 
		// Update the button to be mute/unmute
		if (mediaPlayer.muted) changeButtonType(muteBtn, 'unmute');
		else changeButtonType(muteBtn, 'mute');
	}, false);	
	mediaPlayer.addEventListener('ended', function() { this.pause(); }, false);	

	// mediaPlayer.addEventListener('loadeddata',segment_buttons);

}

function togglePlayPause() {

    var getIcon = document.getElementById('transportIcon');

    if (getIcon.classList.contains('fa-play')) {
        getIcon.classList.remove('fa-play');
        getIcon.classList.add('fa-pause');
        mediaPlayer.play();
    } else {
        getIcon.classList.remove('fa-pause');
        getIcon.classList.add('fa-play');
        mediaPlayer.pause();
    }
}

function changePos(event) {
    var box = progressBar.getBoundingClientRect();
    var pos = (event.pageX - box.left) / box.width;
    mediaPlayer.currentTime = (pos * mediaPlayer.duration);

}

// Stop the current media from playing, and return it to the start position
function stopPlayer() {
	mediaPlayer.pause();
	mediaPlayer.currentTime = 0;
    var getIcon = document.getElementById('transportIcon');

    if (getIcon.classList.contains('fa-play')) {
        getIcon.classList.remove('fa-play');
        getIcon.classList.add('fa-play');
    } else {
        getIcon.classList.remove('fa-pause');
        getIcon.classList.add('fa-play');
    }
}

// Changes the volume on the media player
function changeVolume(direction) {
	if (direction === '+') mediaPlayer.volume += mediaPlayer.volume == 1 ? 0 : 0.1;
	else mediaPlayer.volume -= (mediaPlayer.volume == 0 ? 0 : 0.1);
	mediaPlayer.volume = parseFloat(mediaPlayer.volume).toFixed(1);
}

// Replays the media currently loaded in the player
function replayMedia() {
	resetPlayer();
	mediaPlayer.play();
}

// Update the progress bar
function updateProgressBar() {
	// Work out how much of the media has played via the duration and currentTime parameters
	var percentage = Math.floor((100 / mediaPlayer.duration) * mediaPlayer.currentTime);
	// Update the progress bar's value
	progressBar.value = percentage;
	// Update the progress bar's text (for browsers that don't support the progress element)
	// progressBar.innerHTML = percentage + '% played';
	$("#dynamic")
      .css("width", percentage + "%")
      .attr("aria-valuenow", percentage);
      // .text(percentage + "% Complete");
     var div= document.getElementById('showTime');

     // div.innerHTML=(mediaPlayer.currentTime).toFixed(1)+"ms";

}

// Updates a button's title, innerHTML and CSS class to a certain value
function changeButtonType(btn, value) {
	btn.title = value;
	// btn.innerHTML = value;
	// btn.className = value;

    var getIcon = document.getElementById('transportIcon');
    
    if(value=='pause'){
    if (getIcon.classList.contains('fa-play')) {
        getIcon.classList.remove('fa-play');
        getIcon.classList.add('fa-pause');
    } else {
        getIcon.classList.remove('fa-pause');
        getIcon.classList.add('fa-pause');
    }
   }

    if(value=='play'){
    if (getIcon.classList.contains('fa-play')) {
        getIcon.classList.remove('fa-play');
        getIcon.classList.add('fa-play');
    } else {
        getIcon.classList.remove('fa-pause');
        getIcon.classList.add('fa-play');
    }
   }





}

// Loads a video item into the media player
function loadVideo() {
	for (var i = 0; i < arguments.length; i++) {
		var file = arguments[i].split('.');
		var ext = file[file.length - 1];
		// Check if this media can be played
		if (canPlayVideo(ext)) {
			// Reset the player, change the source file and load it
			resetPlayer();
			mediaPlayer.src = arguments[i];
			mediaPlayer.load();
			break;
		}
	}
}

// Checks if the browser can play this particular type of file or not
function canPlayVideo(ext) {
	var ableToPlay = mediaPlayer.canPlayType('video/' + ext);
	if (ableToPlay == '') return false;
	else return true;
}

// Resets the media player
function resetPlayer() {
    document.getElementById('media-video').removeAttribute('src');
	// Reset the progress bar to 0
	progressBar.value = 0;
	// Move the media back to the start
	mediaPlayer.currentTime = 0;
	// Ensure that the play pause button is set as 'play'
	// changeButtonType(playPauseBtn, 'play');
	var getIcon = document.getElementById('transportIcon');

    if (getIcon.classList.contains('fa-play')) {
        getIcon.classList.remove('fa-play');
        getIcon.classList.add('fa-pause');
    } else {
        getIcon.classList.remove('fa-pause');
        getIcon.classList.add('fa-pause');
    }
}




function segment_buttons(start,end,explanations,associations,flag){

    var elmnt = document.getElementById("progress-bar");
    var w= elmnt.offsetWidth;
    var h= 50;

    var data=[];
    var position=[];
    var mPlayer = document.getElementById("media-video");
    console.log(mPlayer.duration);

    for(var i=0;i<start.length;i++){
        var obj={};
        var percentage = Math.floor((100 / mPlayer.duration) * (start[i]));
        position[i]=(percentage/100)*w;
        console.log(position[i]);
        percentage = Math.floor((100 / mPlayer.duration) * (end[i]));
        temp=(percentage/100)*w;
        width=temp-position[i];

        obj.pos=position[i];
        obj.width=width;
        obj.start=start[i];
        obj.end=end[i];
        data.push(obj);
    }


    var svg= d3.select("#segment")
                .append("svg")
                .attr("width",w)
                .attr("height",h)



    //container for all buttons
    var allButtons= svg.append("g")
                        .attr("id","allButtons");

    //colors for different button states
    var defaultColor= "#8aabff";
    var hoverColor= "#3a6cec";
    var pressedColor= "#001e63";
    var doubleColor="#80002a";

    //groups for each button (which will hold a rect and text)
    var buttonGroups= allButtons.selectAll("g.button")
        .data(data)
        .enter()
        .append("g")
        .attr("class","button")
        .style("cursor","pointer")
        .on("click",function(d,i) {
            d3.selectAll('image').attr("width","16").attr("height","16");
            updateButtonColors(d3.select(this), d3.select(this.parentNode));
            change_segment(d.start,d.end,explanations[i],associations[i],flag);
            // d3.select("#numberToggle").text(i+1)
        })
        .on("mouseover", function() {
            flag=false;
            if ((d3.select(this).select("rect").attr("fill") != pressedColor)){
                d3.select(this)
                    .select("rect")
                    .attr("fill",hoverColor);
            }
        })
        .on("mouseout", function() {
            if ((d3.select(this).select("rect").attr("fill") != pressedColor)) {
                d3.select(this)
                    .select("rect")
                    .attr("fill",defaultColor);
            }
        });


    loadData(explanations[0],associations[0]);
    mPlayer.pause();

    var bHeight= 50; //button height
    // var bSpace= 10; //space between buttons
    // var x0= 20; //x offset
    var y0= 0; //y offset


    var Rect_buttons = buttonGroups.append("rect")
                .attr("class","buttonRect")
                .attr("width",function(d){return d.width;})
                .attr("height",bHeight)
                .attr("x",function(d) {return d.pos;})
                .attr("y",y0)
                .attr("rx",3) //rx and ry give the buttons rounded corners
                .attr("ry",3)
                .attr("fill",function(d,i) {
                    // The first button is always pressed!
                    return (i!=0) ? defaultColor: pressedColor;
                });



    function updateButtonColors(button, parent) {
        parent.selectAll("rect")
                .attr("fill",defaultColor);

        button.select("rect")
                .attr("fill",pressedColor)
    }

    function updateButtonColors2(button, parent) {
        parent.selectAll("rect")
                .attr("fill",defaultColor);

        button.select("rect")
                .attr("fill",doubleColor)
    }


    // if (d3.select("#checkbox0").property("checked") == false) {
    createDropDownForNoSegmentConditions(data, explanations, associations);
    // }

    if (!isNoExplanationCondition())
        mediaPlayer.currentTime=start[0];
}

function isNoExplanationCondition() {
    var vid = d3.select("#checkbox0").property("checked");
    var comp = d3.select("#checkbox1").property("checked");
    var score = d3.select("#checkbox2").property("checked");

    if (!vid && !comp && !score)
        return true;
    else
        return false;
}

function change_segment(time,end,explanations,associations,flag){
    var t1=0;
    var t2=0;
    var t3=0;
    var t4=0;
    var t=0;
    var timer_return_value=false;
    var vid=document.getElementById("media-video");
    // console.log(vid.currentTime);
    // console.log(time);
    // console.log(end);

    t1=(time-Math.floor(time))*100;
    t2=Math.floor(time)*60;
    t2=t2+t1;

    console.log(vid.currentTime);
    console.log(time);
    console.log(t2);

    // if(time<1)
    // {
    // vid.currentTime=time*100;

    // }
    // else
    // {
    //   vid.currentTime=t2;

    // }
    vid.currentTime=time;

      // vid.play();
    vid.pause();

    // t=d3.timer(timeOut);

     function timeOut(){

      // t3=(end-Math.floor(end))*100;
      // t4=Math.floor(end)*60;
      // t4=t3+t4;
       t4=end*100;
      // console.log(vid.currentTime);
      // console.log(end);
      // console.log(t4);

      // var time_temp=vid.currentTime;

      if((vid.currentTime) >= end){
        vid.pause();
        // t.stop();
        timer_return_value=true;
      }

      return timer_return_value;
    };

      // t.restart(timeOut);

                  // if(flag>0) {
                clear_list(flag);
                loadData(explanations, associations);


            // }
            // else {
            //     loadData(explanations, associations);

            // }
            // flag++;

}

function loop_segment(time,end)
{
var t1=0;
var t2=0;
var t3=0;
var t4=0;
var t=0;
var timer_return_value=false;
var vid=document.getElementById("media-video");
console.log(vid.currentTime);
console.log(time);
console.log(end);

t1=(time-Math.floor(time))*100;
t2=Math.floor(time)*60;
t2=t2+t1;

console.log(t2);

// if(time<1)
// {
// vid.currentTime=time*100;

// }
// else
// {
//   vid.currentTime=t2;
 
// }

vid.currentTime=time;

 // vid.play();
vid.pause();
t=d3.timer(timeOut);

 function timeOut(){

  // d3.timerFlush();

  // t3=(end-Math.floor(end))*100;
  // t4=Math.floor(end)*60;
  // t4=t3+t4;
   t4=end*100;

  // var time_temp=vid.currentTime;
   
   if(!flag) t.stop();
  if((vid.currentTime) >= end){

    // vid.pause();  
    // t.stop();
    if(flag)
    {
    vid.currentTime=time;
    // vid.play();
    // d3.timerFlush();
    t.restart(timeOut);

    // loop_segment(time,end);
    // t.stop();
    }

    else
    {
    	vid.pause();
    	t.stop();
    }
    // t.restart(timeOut);
    timer_return_value=true;
  }
   
  return timer_return_value;
};



}

function createDropDownForNoSegmentConditions(dataToChangeTime, explanations, associations) {

    d3.select("#explanation-set-div").html("");
    var mainDiv =
        d3.select("#explanation-set-div")
            .append("div")
            // .attr("id", "explanation-set")
            .classed("col-md-12", true);

    mainDiv.selectAll("div")
        .data(explanations).enter()
        .append("div")
            .style("padding", "5px")
            .classed("col-md-2 explanation-set", true)
        .append("a")
            .classed("col-md-12 explanation-select-item", true)
            .on("click", function (d,i) {
                clear_list();
                loadData(explanations[i], associations[i]);
                d3.selectAll(".explanation-set > a").classed("explanation-select-item-active", false);
                d3.select(this).classed("explanation-select-item-active", true);
                d3.select("#explanation-set-div").node().scrollTop = 0;
                mediaPlayer.currentTime=dataToChangeTime[i].start;
            })
            .html(function (d, i) {
                return "Set " + (i+1);
            });

    // The first button should always be selected!
    d3.select(".explanation-set > a").classed("explanation-select-item-active", true);
}

function clear_segment(){
    d3.select('svg').remove();
}

function clear_explanation_sets() {
    d3.select("#explanation-set-div").html("");
}

async function EnableDisableVideoControls() {
    try {
        if(mediaPlayer !== undefined || mediaPlayer !== null){
            if((mediaPlayer.src === null || mediaPlayer.src === undefined || mediaPlayer.src === "")){

                ['play-pause-button','stop-button','replay-button'].forEach(element => {
                    document.getElementById(element).disabled = true;
                });
            }
            else{
                ['play-pause-button','stop-button','replay-button'].forEach(element => {
                    document.getElementById(element).disabled = false;
                });
            }
        }
        if(mediaPlayer.tagName !== "VIDEO"){
            document.getElementById('media-controls').style.display = 'none';
        }
        else{
            document.getElementById('media-controls').style.display = 'block';
        }
    }
    catch(err) {

    }
    await delay(200);
    EnableDisableVideoControls();
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}