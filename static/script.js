const sendPromptButton = document.getElementById("send");
const apiKeyInput = document.getElementById("apikey");
const promptInput = document.getElementById("prompt");
const responseArea = document.getElementById("response");
const sourcesArea = document.getElementById("sources");


sendPromptButton.addEventListener("click", async () => {
    responseArea.innerHTML = "Procesando... puede tardar en el entorno de un minuto.";
    sourcesArea.innerHTML = "";

    const apiKey = apiKeyInput.value;
    const prompt = promptInput.value;

    if (apiKey === "") {
        responseArea.innerHTML = "INGRESE UNA APIKEY DE OPENAI. Tutorial: https://blog.nubecolectiva.com/como-obtener-una-api-key-de-openai-chatgpt/"
    } else {
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
    }

    const data = await response.json();
    console.log("--- RESPONSE ---");
    console.log(data);

    responseArea.innerHTML = data.final_answer;
    sourcesArea.innerHTML = JSON.stringify(data.thought_process.sources);
})
