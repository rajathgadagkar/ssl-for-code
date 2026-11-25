import * as http from 'http'
import * as vscode from 'vscode'

interface AutocompleteResponse {
	success: boolean
	prediction: string[]
	probs: number[]
	latency_ms?: number
}

function requestCompletions(
	host: string,
	port: number,
	prompt: string
): Promise<AutocompleteResponse> {
	const payload = JSON.stringify({ prompt })

	return new Promise<AutocompleteResponse>((resolve, reject) => {
		let responseData = ''
		const req = http.request(
			{
				hostname: host,
				port,
				path: '/autocomplete',
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Content-Length': Buffer.byteLength(payload, 'utf8'),
				},
				timeout: 3000,
			},
			(res) => {
				res.on('data', (chunk) => {
					responseData += chunk
				})
				res.on('end', () => {
					try {
						resolve(JSON.parse(responseData) as AutocompleteResponse)
					} catch (err) {
						reject(err)
					}
				})
			}
		)

		req.on('error', (error) => reject(error))
		req.on('timeout', () => {
			req.destroy()
			reject(new Error('Autocomplete request timed out'))
		})
		req.write(payload)
		req.end()
	})
}

function getPrompt(document: vscode.TextDocument, position: vscode.Position, contextLines: number): string {
	const start = Math.max(0, position.line - contextLines)
	let text = ''
	for (let i = start; i < position.line; ++i) {
		text += document.lineAt(i).text + '\n'
	}
	const line = document.lineAt(position).text
	text += line.slice(0, position.character)
	return text
}

function hasNewLine(predictions: string[]): boolean[] {
	return predictions.map((p) => p.includes('\n'))
}

function removeNewLine(predictions: string[]): string[] {
	return predictions.map((p) => {
		const nl = p.indexOf('\n')
		return nl !== -1 ? p.slice(0, nl) : p
	})
}

function trimRight(predictions: string[]): string[] {
	return predictions.map((p) => p.trimEnd())
}

function removeDuplicates(predictions: string[]): string[] {
	const seen = new Set<string>()
	const res: string[] = []
	for (const p of predictions) {
		if (p !== '' && !seen.has(p)) {
			seen.add(p)
			res.push(p)
		}
	}
	return res
}

function removeSuffix(
	predictions: string[],
	document: vscode.TextDocument,
	position: vscode.Position
): string[] {
	const line = document.lineAt(position).text
	const trailing = line.slice(position.character)
	if (!trailing) {
		return predictions.filter((p) => p !== '')
	}

	const res: string[] = []
	for (let p of predictions) {
		const suffixIdx = p.indexOf(trailing[0])
		if (suffixIdx !== -1) {
			p = p.slice(0, suffixIdx)
		}
		if (p !== '') {
			res.push(p)
		}
	}
	return res
}

function addPrefix(prefix: string, predictions: string[]): string[] {
	return predictions.map((p) => prefix + p)
}

function buildCompletionItems(predictions: string[], nlFlags: boolean[]): vscode.CompletionItem[] {
	return predictions.map((label, i) => {
		const item = new vscode.CompletionItem(label, vscode.CompletionItemKind.Text)
		item.detail = 'SSL-For-Code'
		if (!nlFlags[i]) {
			item.command = {
				command: 'editor.action.triggerSuggest',
				title: 'Re-trigger completions...',
			}
		}
		return item
	})
}

export function activate(context: vscode.ExtensionContext): void {
	const provider = vscode.languages.registerCompletionItemProvider('python', {
		async provideCompletionItems(
			document: vscode.TextDocument,
			position: vscode.Position
		): Promise<vscode.CompletionItem[]> {
			const config = vscode.workspace.getConfiguration('sslForCode')
			const serverHost = config.get<string>('serverHost', 'localhost')
			const serverPort = config.get<number>('serverPort', 5000)
			const contextLines = config.get<number>('contextLines', 20)
			const confidenceRatio = config.get<number>('confidenceRatio', 4.0)

			const prompt = getPrompt(document, position, contextLines)
			const startedAt = Date.now()

			let response: AutocompleteResponse
			try {
				response = await requestCompletions(serverHost, serverPort, prompt)
			} catch {
				return []
			}

			if (!response.success || !Array.isArray(response.prediction)) {
				return []
			}

			let predictions: string[] = response.prediction
			const probs: number[] = response.probs ?? []

			for (let i = 0; i < probs.length - 1; ++i) {
				if (probs[i] > probs[i + 1] * confidenceRatio) {
					predictions = predictions.slice(0, i + 1)
					break
				}
			}

			const nlFlags = hasNewLine(predictions)
			predictions = removeNewLine(predictions)
			predictions = removeSuffix(predictions, document, position)
			predictions = trimRight(predictions)
			predictions = removeDuplicates(predictions)

			if (predictions.length === 0) {
				if (nlFlags.length > 0 && nlFlags[0]) {
					const newlineItem = new vscode.CompletionItem('\n', vscode.CompletionItemKind.Text)
					newlineItem.command = {
						command: 'editor.action.triggerSuggest',
						title: 'Re-trigger completions...',
					}
					return [newlineItem]
				}
				return []
			}

			const range = document.getWordRangeAtPosition(position)
			if (range) {
				const line = document.lineAt(position).text
				const wordPrefix = line.substring(range.start.character, position.character)
				predictions = addPrefix(wordPrefix, predictions)
			}

			console.log(`SSL-For-Code fetched ${predictions.length} suggestions in ${Date.now() - startedAt}ms`)
			return buildCompletionItems(predictions, nlFlags)
		},
	})

	context.subscriptions.push(provider)
}

export function deactivate(): void {
	// No background resources to clean up.
}
