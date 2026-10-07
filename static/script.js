/* =========================================================
   DEEPSHIELD - MAIN JAVASCRIPT
   ========================================================= */


/* ---------- PAGE READY ---------- */

document.addEventListener("DOMContentLoaded", function () {

    setupAudioUpload();

    setupTranscriptCounter();

    setupAnalysisForm();

    setupNavigation();

});


/* =========================================================
   AUDIO UPLOAD
   ========================================================= */

function setupAudioUpload() {

    const audioInput =
        document.getElementById("audio");

    const fileName =
        document.getElementById("fileName");


    if (!audioInput) {
        return;
    }


    audioInput.addEventListener("change", function () {

        if (!this.files || this.files.length === 0) {

            if (fileName) {
                fileName.textContent = "";
            }

            return;
        }


        const file = this.files[0];


        if (fileName) {

            fileName.textContent =
                "Selected: " + file.name;

        }

    });

}


/* =========================================================
   TRANSCRIPT CHARACTER COUNTER
   ========================================================= */

function setupTranscriptCounter() {

    const transcript =
        document.getElementById("transcript");

    const characterCount =
        document.getElementById("characterCount");


    if (!transcript || !characterCount) {
        return;
    }


    function updateCount() {

        const count =
            transcript.value.length;

        characterCount.textContent =
            count + " characters";

    }


    transcript.addEventListener(
        "input",
        updateCount
    );


    updateCount();

}


/* =========================================================
   ANALYSIS FORM
   ========================================================= */

function setupAnalysisForm() {

    const form =
        document.getElementById("analysisForm");

    const button =
        document.getElementById("analyzeButton");

    const loading =
        document.getElementById("loadingMessage");


    if (!form) {
        return;
    }


    form.addEventListener("submit", function (event) {

        const audioInput =
            document.getElementById("audio");

        const transcript =
            document.getElementById("transcript");


        const hasAudio =
            audioInput &&
            audioInput.files &&
            audioInput.files.length > 0;


        const hasTranscript =
            transcript &&
            transcript.value.trim().length > 0;


        /*
         * Do not allow completely empty analysis.
         */
        if (!hasAudio && !hasTranscript) {

            event.preventDefault();

            alert(
                "Please upload an audio file or enter a conversation transcript."
            );

            return;

        }


        /*
         * Show loading state.
         */
        if (button) {

            button.disabled = true;

            button.innerHTML =
                "⏳ Analyzing...";

        }


        if (loading) {

            loading.style.display =
                "flex";

        }

    });

}


/* =========================================================
   SIDEBAR NAVIGATION
   ========================================================= */

function setupNavigation() {

    const links =
        document.querySelectorAll(".nav-item");


    links.forEach(function (link) {

        link.addEventListener(
            "click",
            function () {

                /*
                 * Do not interfere with normal
                 * Flask navigation.
                 */
            }
        );

    });

}


/* =========================================================
   UTILITY: FORMAT FILE SIZE
   ========================================================= */

function formatFileSize(bytes) {

    if (!bytes || bytes <= 0) {
        return "0 B";
    }


    const units = [
        "B",
        "KB",
        "MB",
        "GB"
    ];


    const index =
        Math.floor(
            Math.log(bytes) /
            Math.log(1024)
        );


    const size =
        bytes /
        Math.pow(1024, index);


    return (
        size.toFixed(1) +
        " " +
        units[index]
    );

}


/* =========================================================
   UTILITY: FORMAT DATE
   ========================================================= */

function formatDate(dateValue) {

    const date =
        new Date(dateValue);


    if (isNaN(date.getTime())) {
        return dateValue;
    }


    return date.toLocaleString(
        "en-IN",
        {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        }
    );

}


/* =========================================================
   RISK COLOR HELPER
   ========================================================= */

function getRiskClass(score) {

    score =
        Number(score) || 0;


    if (score >= 70) {
        return "high";
    }


    if (score >= 40) {
        return "medium";
    }


    return "low";

}


/* =========================================================
   PREVENT DOUBLE SUBMISSION
   ========================================================= */

window.addEventListener(
    "beforeunload",
    function () {

        /*
         * Allows the browser to finish normal
         * form navigation without extra actions.
         */

    }
);