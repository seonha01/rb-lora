window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})


function copyBibtex(event) {
    const btn = event.currentTarget;
    const icon = btn.querySelector("i");
    const text = document.getElementById("bibtex-block").innerText;

    function showSuccess() {
        icon.classList.remove("fa-copy");
        icon.classList.add("fa-check");

        setTimeout(() => {
            icon.classList.remove("fa-check");
            icon.classList.add("fa-copy");
        }, 1500);
    }

    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(showSuccess);
    } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);

        showSuccess();
    }
}
