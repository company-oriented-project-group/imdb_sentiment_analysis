// assets/sentiment_meter_animation.js
// setInterval(updateMeter, 1000);

function updateMeter() {
    // var sliderValue = document.getElementById('sentiment-slider').value;
    // var meter = document.getElementById('sentiment-meter');
    var soundEffect = document.getElementById('sound-effect');

    // meter.style.width = sliderValue + '%';

    //test
    sliderValue = 99

    if (sliderValue < 30) {
        soundEffect.src = '/assets/negative.mp3';
    } else if (sliderValue > 70) {
        soundEffect.src = '/assets/positive.mp3';
    } else {
        soundEffect.src = '/assets/neutral.mp3';
    }

    soundEffect.play();
}
