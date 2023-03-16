const sendPromptButton = document.getElementById("send");
const apiKeyInput = document.getElementById("apikey");
const promptInput = document.getElementById("prompt");
const responseArea = document.getElementById("response");
const sourcesArea = document.getElementById("sources");


sendPromptButton.addEventListener("click", async () => {
    const apiKey = apiKeyInput.value;
    const prompt = promptInput.value;
    const response = await fetch("/send_question", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            api_key: apiKey,
            prompt,
            law_filter: null
        })
    });

    const data = await response.json();
    console.log("--- RESPONSE ---");
    console.log(data);

    responseArea.innerHTML = data.final_answer;
    sourcesArea.innerHTML = JSON.stringify(data.thought_process.sources);
})
