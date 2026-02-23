import { useState, useRef, useEffect } from "react";
import UploadModal from "./UploadModal";


interface Source {
  document: string;
  page: number;
  score: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

const API_URL = "http://localhost:8000";


export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const query = input.trim();
    if (!query || isLoading) return;

    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error(`Błąd serwera: ${res.status}`);

      const data = await res.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Błąd połączenia z serwerem.`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-stone-50">
      {/* Header */}
      <header className="border-b border-stone-200 bg-white px-6 py-4">
        <h1 className="text-lg font-semibold text-stone-800 tracking-tight">
          Chat
        </h1>
        <p className="text-sm text-stone-400">
          Zadawaj pytania na podstawie dokumentów
        </p>
      <button
          onClick={() => setIsUploadOpen(true)}
          className="rounded-xl border border-stone-200 bg-stone-50 px-4 py-2 text-sm text-stone-600 hover:bg-stone-100 transition-colors"
        >
          + Dodaj PDF
        </button>
      </header>

      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
      />

      {/* Message list */}
      <main className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-stone-300 text-sm">
              Zadaj pytanie, aby rozpocząć rozmowę
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-stone-800 text-white"
                  : "bg-white text-stone-700 border border-stone-200"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {/* Sources shown for assistant */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-stone-100 flex flex-wrap gap-2">
                  {[...new Map(
                    msg.sources.map((src) => [`${src.document}:${src.page}`, src])
                  ).values()].map((src, j) => (
                    <span
                      key={j}
                      className="text-xs text-stone-400 bg-stone-50 border border-stone-100 rounded-full px-2 py-0.5"
                    >
                      {src.document}, str. {src.page}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-stone-200 rounded-2xl px-4 py-3">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce" />
                <div className="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce [animation-delay:0.15s]" />
                <div className="w-1.5 h-1.5 bg-stone-300 rounded-full animate-bounce [animation-delay:0.3s]" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Message input */}
      <footer className="border-t border-stone-200 bg-white px-6 py-4">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Zadaj pytanie..."
            disabled={isLoading}
            className="flex-1 rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-800 placeholder-stone-300 outline-none focus:border-stone-400 focus:ring-1 focus:ring-stone-400 transition-colors disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="rounded-xl bg-stone-800 px-5 py-3 text-sm font-medium text-white hover:bg-stone-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Wyślij
          </button>
        </div>
      </footer>
    </div>
  );
}