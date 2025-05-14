document.addEventListener("DOMContentLoaded", function () {
  emailjs.init("YOUR_EMAILJS_USER_ID"); // Replace with your actual User ID

  const contactForm = document.querySelector(".contact-form");
  if (contactForm) {
    contactForm.addEventListener("submit", function (event) {
      event.preventDefault();

      const name = document.querySelector(
        'input[placeholder="Your Name"]'
      ).value;
      const email = document.querySelector(
        'input[placeholder="Your Email"]'
      ).value;
      const message = document.querySelector(
        'textarea[placeholder="Your Message"]'
      ).value;

      const formData = {
        name: name,
        email: email,
        message: message,
      };

      emailjs
        .send("YOUR_EMAILJS_SERVICE_ID", "YOUR_EMAILJS_TEMPLATE_ID", formData) // Replace with your Service ID and Template ID
        .then(
          function (response) {
            console.log("SUCCESS!", response.status, response.text);

            // Show success toast
            const successToast = document.getElementById("successToast");
            if (successToast) {
              const toast = new bootstrap.Toast(successToast);
              toast.show();
            }

            if (contactForm) {
              contactForm.reset();
            }
          },
          function (error) {
            console.log("FAILED...", error);

            // Show failure toast
            const failureToast = document.getElementById("failureToast");
            if (failureToast) {
              const toast = new bootstrap.Toast(failureToast);
              toast.show();
            }
          }
        );
    });
  } else {
    console.error("Contact form not found!");
  }
});
