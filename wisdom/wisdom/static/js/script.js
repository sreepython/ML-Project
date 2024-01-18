document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendButton = document.getElementById('send-button');

    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();
        sendMessage();
    });

    sendButton.addEventListener('click', function() {
        sendMessage();
    });

    function sendMessage() {
        const userMessage = userInput.value.trim();
        if (userMessage !== '') {
            appendMessage('You', userMessage);
            userInput.value = '';

            // Show loading spinner while waiting for the response
            showLoadingSpinner();

            // Send user input to the server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'user_input=' + encodeURIComponent(userMessage),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner when the response is received
                hideLoadingSpinner();

                const serverResponse = data.response;
                appendMessage('Server', serverResponse);
            })
            .catch(error => {
                // Handle errors, if any
                console.error('Error:', error);
                hideLoadingSpinner();
            });
        }
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showLoadingSpinner() {
        // Show loading spinner
        document.getElementById('loading-spinner').style.display = 'block';
    }

    function hideLoadingSpinner() {
        // Hide loading spinner
        document.getElementById('loading-spinner').style.display = 'none';
    }
});
