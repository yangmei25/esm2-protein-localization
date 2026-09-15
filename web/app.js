const sequence = document.querySelector("#sequence");
const proteinId = document.querySelector("#protein-id");
const predictButton = document.querySelector("#predict");
const exampleButton = document.querySelector("#example");
const error = document.querySelector("#error");
const result = document.querySelector("#result");
const exampleSequence = "MKTIIALSYIFCLVFADYKDDDDK";

function normalizedLength() {
  return sequence.value.replace(/\s/g, "").length;
}
function percent(value) { return `${(100 * value).toFixed(1)}%`; }
sequence.addEventListener("input", () => {
  document.querySelector("#sequence-length").textContent = `${normalizedLength()} residues`;
});
exampleButton.addEventListener("click", () => {
  proteinId.value = "example_protein";
  sequence.value = exampleSequence;
  sequence.dispatchEvent(new Event("input"));
});
predictButton.addEventListener("click", async () => {
  error.textContent = "";
  result.classList.add("hidden");
  predictButton.disabled = true;
  predictButton.textContent = "Running ESM-2…";
  try {
    const response = await fetch("/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ protein_id: proteinId.value, sequence: sequence.value }),
    });
    const responseText = await response.text();
    let payload;
    try {
      payload = JSON.parse(responseText);
    } catch {
      throw new Error(
        response.ok
          ? "The server returned an invalid response."
          : `Server error (${response.status}). Check the service terminal.`
      );
    }
    if (!response.ok) throw new Error(payload.detail || `Prediction failed (${response.status})`);
    document.querySelector("#predicted-label").textContent = payload.predicted_label;
    const confidence = Math.max(payload.membrane_probability, payload.soluble_probability);
    document.querySelector("#confidence").textContent = `${percent(confidence)} confidence`;
    document.querySelector("#membrane-value").textContent = percent(payload.membrane_probability);
    document.querySelector("#soluble-value").textContent = percent(payload.soluble_probability);
    document.querySelector("#membrane-bar").style.width = percent(payload.membrane_probability);
    document.querySelector("#soluble-bar").style.width = percent(payload.soluble_probability);
    document.querySelector("#model-name").textContent = payload.model_name;
    document.querySelector("#checkpoint-epoch").textContent = payload.checkpoint_epoch ?? "not recorded";
    document.querySelector("#device").textContent = payload.device;
    document.querySelector("#result-length").textContent = `${payload.sequence_length} aa`;
    document.querySelector("#threshold").textContent = payload.threshold.toFixed(2);
    document.querySelector("#windows").textContent = payload.number_of_windows;
    document.querySelector("#top-window").textContent = `${payload.top_window.start}–${payload.top_window.end}`;
    result.classList.remove("hidden");
  } catch (requestError) {
    error.textContent = requestError.message;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict localization";
  }
});
