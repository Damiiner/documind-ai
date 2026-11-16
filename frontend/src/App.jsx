import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

// the only styling: boxes with a border, and long file names that wrap instead of overflowing
const box = { border: "1px solid #ddd", borderRadius: 6, padding: 8, marginBottom: 8 };
const wrap = { overflowWrap: "anywhere" };

export default function App() {
  const [docs, setDocs] = useState([]);
  const [selected, setSelected] = useState(""); // "" means all documents
  const [mode, setMode] = useState("chat"); // "chat" or "quiz"
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  // chat
  const [messages, setMessages] = useState([]); // { role, text, sources }
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  // quiz
  const [quiz, setQuiz] = useState([]);
  const [choices, setChoices] = useState({}); // question index -> chosen option index
  const [checked, setChecked] = useState(false);
  const [quizCount, setQuizCount] = useState(5);
  const [quizLoading, setQuizLoading] = useState(false);

  async function loadDocs() {
    const res = await fetch(`${API}/documents`);
    setDocs(await res.json());
  }

  useEffect(() => {
    loadDocs().catch(() => setError("Cannot reach the API. Is uvicorn running?"));
  }, []);

  async function upload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/documents`, { method: "POST", body: form });
      if (!res.ok) {
        const data = await res.json();
        setError(typeof data.detail === "string" ? data.detail : "Upload failed");
      }
      await loadDocs();
    } catch {
      setError("Upload failed. Is the API running?");
    }
    setUploading(false);
    e.target.value = "";
  }

  async function remove(name) {
    await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (selected === name) setSelected("");
    loadDocs();
  }

  async function ask(e) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    const q = question;
    // the last 4 messages are sent so follow-up questions ("and for SD3.5?") can be understood
    const history = messages.slice(-4).map((m) => ({ role: m.role, text: m.text }));
    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "", sources: [], used: "" }]);
    setQuestion("");
    setLoading(true);
    setError("");

    // change the last message (the assistant answer that is being written)
    const updateLast = (change) =>
      setMessages((m) => m.map((msg, i) => (i === m.length - 1 ? { ...msg, ...change(msg) } : msg)));

    try {
      const res = await fetch(`${API}/ask_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, source: selected || null, history }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // the last piece may be an unfinished line
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === "question") updateLast(() => ({ used: event.text }));
          else if (event.type === "sources") updateLast(() => ({ sources: event.sources }));
          else if (event.type === "token") updateLast((msg) => ({ text: msg.text + event.text }));
          else if (event.type === "error") setError(event.text);
        }
      }
    } catch {
      setError("Request failed. Is the API running?");
    }
    setLoading(false);
  }

  async function makeQuiz() {
    setQuizLoading(true);
    setError("");
    setQuiz([]);
    setChoices({});
    setChecked(false);
    try {
      const res = await fetch(`${API}/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: selected, n: quizCount }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(typeof data.detail === "string" ? data.detail : "Quiz generation failed");
      } else {
        setQuiz(data.questions);
      }
    } catch {
      setError("Request failed. Is the API running?");
    }
    setQuizLoading(false);
  }

  const score = quiz.filter((q, i) => choices[i] === q.answer).length;

  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-start", maxWidth: 1100, margin: "0 auto", padding: 16, fontFamily: "system-ui, sans-serif" }}>
      {/* left: documents */}
      <div data-panel="left" style={{ width: 280, flexShrink: 0 }}>
        <h2>DocuMind AI</h2>
        <label>
          <input type="file" accept=".pdf" onChange={upload} disabled={uploading} />
        </label>
        {uploading && <p>Processing PDF...</p>}

        <h3>Documents</h3>
        {docs.length === 0 && <p>No documents yet.</p>}
        <div style={{ ...box, background: selected === "" ? "#eef3ff" : "white" }}>
          <label style={{ display: "flex", gap: 6 }}>
            <input type="radio" checked={selected === ""} onChange={() => setSelected("")} />
            <span>All documents</span>
          </label>
        </div>
        {docs.map((d) => (
          <div key={d.name} style={{ ...box, background: selected === d.name ? "#eef3ff" : "white" }}>
            <label style={{ display: "flex", gap: 6 }}>
              <input type="radio" checked={selected === d.name} onChange={() => setSelected(d.name)} />
              <span style={wrap}>{d.name}</span>
            </label>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 }}>
              <small>{d.chunks} chunks</small>
              <button onClick={() => remove(d.name)}>delete</button>
            </div>
          </div>
        ))}
      </div>

      {/* right: chat or quiz */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ marginBottom: 12 }}>
          <button onClick={() => setMode("chat")} disabled={mode === "chat"}>
            Chat
          </button>{" "}
          <button onClick={() => setMode("quiz")} disabled={mode === "quiz"}>
            Quiz
          </button>
        </div>

        {error && <p style={{ color: "red" }}>{error}</p>}

        {mode === "chat" && (
          <div>
            {messages.length > 0 && (
              <p>
                <button onClick={() => setMessages([])} disabled={loading}>
                  New chat
                </button>
              </p>
            )}
            {messages.map((m, i) => (
              <div key={i} style={{ ...box, background: m.role === "user" ? "#f1f3f5" : "white", marginBottom: 12 }}>
                <b>{m.role === "user" ? "You" : "DocuMind"}</b>
                {m.used && m.used !== messages[i - 1]?.text && (
                  <div>
                    <small>Searched for: {m.used}</small>
                  </div>
                )}
                <div style={{ whiteSpace: "pre-wrap", ...wrap }}>
                  {m.text}
                  {loading && i === messages.length - 1 && m.role === "assistant" ? "▌" : ""}
                </div>
                {m.sources && m.sources.length > 0 && (
                  <details style={{ marginTop: 6 }}>
                    <summary>Sources ({m.sources.length})</summary>
                    {m.sources.map((s, j) => (
                      <details key={j} style={{ marginLeft: 16 }}>
                        <summary style={wrap}>
                          {s.source}, page {s.page}
                        </summary>
                        <small style={wrap}>{s.text}...</small>
                      </details>
                    ))}
                  </details>
                )}
              </div>
            ))}
            {loading && messages[messages.length - 1]?.text === "" && <p>Thinking...</p>}

            <form onSubmit={ask} style={{ display: "flex", gap: 8 }}>
              <input
                style={{ flex: 1 }}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={selected ? `Ask about ${selected}` : "Ask about your documents"}
              />
              <button type="submit" disabled={loading || docs.length === 0}>
                Ask
              </button>
            </form>
          </div>
        )}

        {mode === "quiz" && (
          <div>
            {selected === "" ? (
              <p>Select one document on the left to generate a quiz from it.</p>
            ) : (
              <div>
                <p>
                  Quiz from <b>{selected}</b>:{" "}
                  <select value={quizCount} onChange={(e) => setQuizCount(Number(e.target.value))}>
                    {[3, 5, 8].map((n) => (
                      <option key={n} value={n}>
                        {n} questions
                      </option>
                    ))}
                  </select>{" "}
                  <button onClick={makeQuiz} disabled={quizLoading}>
                    Generate quiz
                  </button>
                </p>
                {quizLoading && <p>Generating and checking questions... this can take a few minutes.</p>}
              </div>
            )}

            {quiz.map((q, i) => (
              <div key={i} style={{ marginBottom: 16 }}>
                <b>
                  {i + 1}. {q.question}
                </b>
                {q.options.map((opt, j) => {
                  let color = "black";
                  if (checked && j === q.answer) color = "green";
                  else if (checked && choices[i] === j) color = "red";
                  return (
                    <label key={j} style={{ display: "block", color }}>
                      <input
                        type="radio"
                        name={`q${i}`}
                        checked={choices[i] === j}
                        disabled={checked}
                        onChange={() => setChoices({ ...choices, [i]: j })}
                      />{" "}
                      {opt}
                    </label>
                  );
                })}
                {checked && (
                  <small>
                    {q.explanation} (source: page {q.page})
                  </small>
                )}
              </div>
            ))}

            {quiz.length > 0 && !checked && (
              <button onClick={() => setChecked(true)} disabled={Object.keys(choices).length < quiz.length}>
                Check answers
              </button>
            )}
            {checked && (
              <p>
                <b>
                  Score: {score} / {quiz.length}
                </b>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
