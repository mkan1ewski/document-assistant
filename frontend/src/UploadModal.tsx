import { useState, useCallback } from "react";

const API_URL = "http://localhost:8000";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      setResult("Tylko pliki PDF są obsługiwane.");
      return;
    }

    setIsUploading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Błąd serwera: ${res.status}`);

      const data = await res.json();

      if (data.chunks_added === 0) {
        setResult(`Dokument "${data.filename}" znajduje się już w bazie.`);
      } else {
        setResult(`Pomyślnie dodano dokument "${data.filename}".`);
      }

    } catch {
      setResult("Błąd połączenia z serwerem.");
    } finally {
      setIsUploading(false);
    }
  }, []);

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-lg w-full max-w-md mx-4 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-stone-800">
            Dodaj dokument
          </h2>
          <button
            onClick={onClose}
            className="text-stone-400 hover:text-stone-600 text-lg leading-none"
          >
            ✕
          </button>
        </div>

        <label
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`flex flex-col items-center justify-center h-40 rounded-xl border-2 border-dashed cursor-pointer transition-colors ${
            isDragging
              ? "border-stone-400 bg-stone-100"
              : "border-stone-200 bg-stone-50 hover:border-stone-300"
          }`}
        >
          <p className="text-sm text-stone-400">
            {isUploading
              ? "Przetwarzanie..."
              : "Przeciągnij PDF lub kliknij"}
          </p>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            disabled={isUploading}
            className="hidden"
          />
        </label>

        {result && (
          <p className="mt-3 text-sm text-stone-600 text-center">{result}</p>
        )}
      </div>
    </div>
  );
}