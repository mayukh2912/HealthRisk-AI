import Link from "next/link";
export default function NotFound() {
  return <div className="py-20 text-center"><h1 className="text-xl font-semibold text-navy-900">Page not found</h1><Link className="mt-3 inline-block text-teal-700 underline" href="/">Back to dashboard</Link></div>;
}
